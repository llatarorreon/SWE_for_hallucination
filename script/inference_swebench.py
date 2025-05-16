import os
import json
import argparse
import logging
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import time
import datetime
import re
import copy
from tqdm import tqdm

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# might need to change for different models (like Claude/Llama)
SWE_TOOLS_PATH = "swebench_tool_gpt.json"

# Import model configuration
from model_config import (
    get_client,
    get_model_config,
    get_available_models,
)

class ParseError(Exception):
    """Tag parsing error"""

    pass


# Set up logging
def setup_logger(log_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    """Set up logger"""
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"inference_{timestamp}.log"

    # Configure logger
    logger = logging.getLogger("inference")
    logger.setLevel(log_level)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_dataset(dataset_dir: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Load all info.json files from the dataset directory"""
    dataset = []
    logger.info(f"Starting to load dataset: {dataset_dir}")

    # Check if directory exists
    if not dataset_dir.exists():
        logger.error(f"Dataset directory does not exist: {dataset_dir}")
        return []

    # Traverse all JSON files in the directory
    for json_file in dataset_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                info_data = json.load(f)

            # 确保JSON文件已包含转换好的input
            if "input" not in info_data:
                logger.warning(f"Input field not found in {json_file.name}")
                raise ValueError(f"Input field not found in {json_file.name}")

            dataset.append(info_data)
            logger.debug(f"Task loaded: {json_file.name}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {json_file.name}: JSON format error - {e}")
        except Exception as e:
            logger.error(f"Error reading {json_file.name}: {e}")

    logger.info(f"Dataset loading complete, total {len(dataset)} tasks")
    return dataset


def query_model(
    model_name: str,
    messages: List[Dict[str, Any]],
    logger: logging.Logger,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_retries: int = 3
) -> Dict[str, Any]:
    """Query model and handle possible errors, including format parsing errors"""
    model_config = get_model_config(model_name)
    model_id = model_config["model_id"]
    temperature = model_config["temperature"]
    max_tokens = model_config["max_tokens"]

    # Get model-specific client
    try:
        client = get_client(model_name)
    except Exception as e:
        logger.error(f"Error getting client for model {model_name}: {e}")
        return {"model": model_name, "error": str(e)}

    # Log messages information
    logger.debug(f"Number of messages: {len(messages)}")

    current_messages = messages

    # Multiple attempts to call API and parse results
    for retry in range(max_retries):
        try:
            logger.debug(
                f"Starting to query model {model_name}, attempt: {retry+1}/{max_retries}"
            )

            # Send request
            response = client.chat.completions.create(
                model=model_id,
                messages=current_messages,
                tools=tools,
                tool_choice="auto" if tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            assistant_msg = response.choices[0].message.model_dump()
            return {
                "model": model_name,
                "completion": assistant_msg,
                "thinking": assistant_msg.get("content", ""),
                "action": assistant_msg.get("tool_calls") or [],
                "timestamp": datetime.datetime.now().isoformat(),
                "assistant_message": assistant_msg
            }

        except Exception as e:
            if retry < max_retries - 1:
                sleep_time = 2**retry  # Exponential backoff
                logger.warning(
                    f"Error requesting model {model_name}: {e}. Will retry in {sleep_time} seconds..."
                )
                time.sleep(sleep_time)
            else:
                logger.error(
                    f"Failed to request model {model_name}, maximum retry attempts reached: {e}"
                )
                return {
                    "model": model_name,
                    "error": str(e),
                    "timestamp": datetime.datetime.now().isoformat(),
                }


def save_results(
    task_data: Dict[str, Any],
    model_result: Dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Save results to output directory"""
    # Copy original task_data, create result object
    result_data = task_data.copy()

    task_name = result_data["task_name"]
    # Add model results
    result_data["result"] = model_result

    # Build output path, ensure directory exists
    model_dir = output_dir / model_result["model"]
    model_dir.mkdir(parents=True, exist_ok=True)

    # Use original task directory name as part of result filename
    result_file = model_dir / f"{task_name}.json"

    # Save results
    try:
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Results saved to {result_file}")
    except Exception as e:
        logger.error(f"Error saving results to {result_file}: {e}")

    return result_file


def check_result_exists(task_name: str, model: str, result_dir: Path) -> bool:
    """Check if the result file for a specific task already exists"""
    result_file = result_dir / model / f"{task_name}.json"
    return result_file.exists()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test multiple LLM models and save results"
    )
    parser.add_argument(
        "--risk-setting",
        type=str,
        default="unachievable",
        help="Risk setting folder name (e.g., 'unachievable', 'harmful')",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="workarena",
        help="Scenario folder name (e.g., 'workarena', 'shopping')",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="../inferenced_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--log-dir", type=str, default="../logs", help="Directory to save logs"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "gpt-4o-2024-11-20",
            # "gpt-4o-mini",
            # "gemini-2.0-flash",
            # "claude-3-5-sonnet-20240620",
            # "deepseek-r1",
            # "Qwen2.5-7B-Instruct",
            # "llama-3.3-70b-chat",
        ],
        help="List of models to test, available models: "
        + ", ".join(get_available_models()),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode, process only a few samples and output detailed logs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Log level",
    )
    parser.add_argument(
        "--max-parse-retries",
        type=int,
        default=3,
        help="Maximum retry attempts when model output tag parsing fails",
    )
    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()

    # Build dataset path
    dataset_path = f"../dataset_all/{args.risk_setting}/{args.scenario}"

    # Set paths
    dataset_dir = Path(dataset_path)
    result_dir = Path(args.result_dir) / args.risk_setting
    log_dir = Path(args.log_dir)

    # Ensure result directory exists
    result_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(log_dir, log_level)
    logger.info("=" * 50)
    logger.info("Starting inference task")
    logger.info(f"Risk setting: {args.risk_setting}")
    logger.info(f"Scenario: {args.scenario}")
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Result save directory: {result_dir}")
    logger.info(f"Test models: {', '.join(args.models)}")
    logger.info("=" * 50)

    # Load dataset
    dataset = load_dataset(dataset_dir, logger)
    if not dataset:
        logger.error("Dataset is empty, exiting program")
        return

    if args.debug:
        # In debug mode, only process first 3 samples
        dataset = dataset[:3]
        logger.info(f"Debug mode: Only processing {len(dataset)} samples")

    # Run tests for each model
    for model in args.models:
        logger.info(f"\nStarting to test model: {model}")
        model_results = []
        skipped_tasks = 0
        new_tasks = 0
        model_start_time = time.time()

        try:
            # Ensure model result directory exists
            model_dir = result_dir / model
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(SWE_TOOLS_PATH, "r", encoding="utf-8") as f:
                    swe_tools = json.load(f)
            # Process each task
            for task_idx, task_data in enumerate(
                tqdm(dataset, desc=f"Testing {model}")
            ):
                task_name = task_data.get("task_name")
                logger.info(
                    f"Processing task {task_name} ({task_idx+1}/{len(dataset)})"
                )

                # Check if result file already exists
                if check_result_exists(task_name, model, result_dir):
                    logger.info(f"Result file exists, skipping task {task_name}")
                    skipped_tasks += 1
                    model_results.append(model_dir / f"{task_name}.json")
                    continue

                # Use the pre-converted messages data directly
                messages = copy.deepcopy(
                    task_data["input"]
                )  # Create deep copy to avoid modifying original data

                # Query model
                model_result = query_model(
                    model_name=model,
                    messages=messages,
                    logger=logger,
                    tools=swe_tools
                )

                # Save results
                result_file = save_results(task_data, model_result, result_dir, logger)
                model_results.append(result_file)
                new_tasks += 1

                if "error" not in model_result:
                    logger.info(f"Task {task_name} processed successfully")
                else:
                    logger.error(
                        f"Task {task_name} processing failed: {model_result.get('error')}"
                    )

                # Brief pause to avoid API rate limits
                time.sleep(0.5)

        except KeyboardInterrupt:
            logger.warning("User interrupted, ending model test early")
        except Exception as e:
            logger.error(f"Error occurred while testing model {model}: {e}")

        # Calculate and record model test time
        model_time = time.time() - model_start_time
        total_tasks = skipped_tasks + new_tasks
        logger.info(
            f"Model {model} testing complete: {total_tasks} total tasks processed"
        )
        logger.info(f"- Skipped tasks: {skipped_tasks}, New tasks: {new_tasks}")
        logger.info(f"- Total time: {model_time:.2f} seconds")

    logger.info("\nAll model tests complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nUser interrupted, program exiting")
    except Exception as e:
        print(f"Error occurred during program execution: {e}")
