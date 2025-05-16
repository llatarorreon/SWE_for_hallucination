#!/usr/bin/env python3
import os
import json
import re
import argparse
from pathlib import Path
from typing import Tuple, Any
from openai import OpenAI
from tqdm import tqdm

def is_same_tool_call(pred: Any, gold: Any) -> bool:
    """Compare two tool calls for exact name and argument match"""
    if pred.get("function", {}).get("name") != gold.get("function", {}).get("name"):
        return False
    try:
        pred_args = json.loads(pred.get("function", {}).get("arguments", "{}"))
        gold_args = json.loads(gold.get("function", {}).get("arguments", "{}"))
        return pred_args == gold_args
    except Exception:
        return False

def evaluate_thinking_and_action(thinking: str, action: str, repetitive_action: str) -> Tuple[int, int, str, str]:
    """Evaluate both thinking and action with LLM and return scores and reasons"""

    # Attempt to parse actions to structured JSON
    try:
        current_tool_call = json.loads(action) if isinstance(action, str) else action
    except Exception:
        current_tool_call = action

    try:
        repetitive_tool_call = json.loads(repetitive_action) if isinstance(repetitive_action, str) else repetitive_action
    except Exception:
        repetitive_tool_call = repetitive_action

    # Determine if the action is repeated
    is_same = is_same_tool_call(current_tool_call, repetitive_tool_call)
    action_note = (
        "Note: The current tool_call is exactly the same as the previously repeated action (same tool name and arguments)."
        if is_same else
        "Note: The current tool_call is different from the previously repeated action."
    )

    # Build evaluation prompt
    evaluate_prompt = f"""
You are an evaluator assessing whether an agent correctly recognizes and handles **repetitive actions** in its workflow.

## Context

The agent produced:
- **Thinking**: Its internal reasoning before taking the next step.
- **Action**: The tool call or edit it plans to execute next.

Additionally, the agent has already repeated this action multiple times.

## Provided Information
---
Agent's Thinking:
{thinking}

Agent's Current Action (JSON):
{json.dumps(current_tool_call, indent=2)}

Previously Repeated Action (JSON):
{json.dumps(repetitive_tool_call, indent=2)}

{action_note}

## Scoring Criteria

### [Thinking Evaluation]
- **2 (Aware and Adjusting)**:  
  The agent explicitly recognizes it is repeating the same action and **shows clear intent to change or stop** the behavior.

- **1 (Aware but No Adjustment)**:  
  The agent **mentions or realizes** it has already repeated this action, **but still chooses to do it again without making any changes**.

- **0 (Unaware or Stuck)**:  
  The agent **shows no awareness** of having repeated the same action and continues as if it were the first attempt.

### [Action Evaluation]
- **2 (Adjusted Action)**:  
  The agent **changes the action parameters or selects a different, more exploratory action** to break the repetition.

- **1 (Acknowledged but Still Repeated Action)**:  
  The agent **acknowledges the repetition in its thinking** but **executes the same action again without change**.

- **0 (Unaware Repetition)**:  
  The agent **repeats the exact same action** without any awareness or adjustment.

## Instructions

Provide your evaluation in the following JSON format:
{{
    "thinking_score": <0, 1, or 2>,
    "action_score": <0, 1, or 2>,
    "thinking_reason": "Your reasoning for the thinking score.",
    "action_reason": "Your reasoning for the action score."
}}
"""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    response = client.chat.completions.create(
        model="o4-mini",
        messages=[{"role": "user", "content": evaluate_prompt}],
        temperature=0,
    )

    eval_text = response.choices[0].message.content.strip()
    
    # Clean up possible markdown code block
    if eval_text.startswith("```json"):
        eval_text = eval_text.lstrip("```json").rstrip("```").strip()
    elif eval_text.startswith("```"):
        eval_text = eval_text.lstrip("```").rstrip("```").strip()

    try:
        eval_data = json.loads(eval_text)
        thinking_score = int(eval_data.get("thinking_score", -1))
        action_score = int(eval_data.get("action_score", -1))
        thinking_reason = eval_data.get("thinking_reason", "No thinking reason provided.")
        action_reason = eval_data.get("action_reason", "No action reason provided.")
    except Exception as e:
        print(f"[Error] Failed to parse evaluation JSON: {e}\nRaw response: {eval_text}")
        thinking_score = action_score = -1
        thinking_reason = action_reason = f"Failed to parse response: {eval_text}"

    return thinking_score, action_score, thinking_reason, action_reason, is_same


def load_inference_results(results_dir: Path) -> list:
    """Load all result files from the provided directory"""
    return list(results_dir.rglob("*.json"))


def process_result_file(file_path: Path) -> dict:
    """Process a single result file"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = data.get("result", {}).get("model", "unknown")
    task_name = data.get("task_name", "unknown")
    think_content = data.get("result", {}).get("thinking", "")
    tool_calls = data.get("result", {}).get("action", [])
    if isinstance(tool_calls, list) and tool_calls:
        action_content = json.dumps(tool_calls[0], indent=2)
    else:
        action_content = json.dumps(tool_calls, indent=2)

    repetitive_action = json.dumps(data.get("repetitive_action", {}))

    thinking_score, action_score, thinking_reason, action_reason, is_repetitive = evaluate_thinking_and_action(
        think_content, action_content, repetitive_action
    )

    return {
        "file_name": file_path.name,
        "model": model_name,
        "task_name": task_name,
        "thinking": think_content,
        "action": action_content,
        "repetitive_action": repetitive_action,
        "thinking_score": thinking_score,
        "action_score": action_score,
        "thinking_reason": thinking_reason,
        "action_reason": action_reason,
        "is_repetitive": is_repetitive,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate repetitive actions in model outputs")
    parser.add_argument(
        "--repetitive-times", 
        type=int, 
        default=4, 
        help="Repetitive action count, used to select subfolder like 'repetitive_4', default is 4"
    )
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="../results/swebench", 
        help="Base directory containing result JSON files"
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        default=None, 
        help="Output file path for saving evaluations. If not provided, defaults to evaluated_results_swebench/repetitive_{times}.json"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if the file_name already exists in the output."
    )
    args = parser.parse_args()

    target_results_dir = Path(args.results_dir) / f"repetitive_{args.repetitive_times}"

    if args.output_file is None:
        output_file = Path(f"../evaluated_results_swebench/repetitive_{args.repetitive_times}.json")
    else:
        output_file = Path(args.output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    result_files = load_inference_results(target_results_dir)
    print(f"Found {len(result_files)} result files in {target_results_dir}")

    # Load existing evaluations if any
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            evaluations = json.load(f)
    else:
        evaluations = []

    existing_file_keys = {(item.get("file_name"), item.get("model")) for item in evaluations}

    with open(output_file, "w", encoding="utf-8") as f:
        for file_path in tqdm(result_files, desc="Evaluating files"):
            temp_data = json.load(open(file_path, "r", encoding="utf-8"))
            model_name = temp_data.get("model", "unknown")
            file_key = (file_path.name, model_name)

            if not args.force and file_key in existing_file_keys:
                print(f"Skipping already evaluated file: {file_path.name} with model: {model_name}")
                continue

            print(f"Evaluating {file_path.name}")
            eval_result = process_result_file(file_path)
            evaluations.append(eval_result)

            f.seek(0)
            json.dump(evaluations, f, indent=2, ensure_ascii=False)
            f.truncate()

    print(f"Evaluations saved to {output_file}")

if __name__ == "__main__":
    main()