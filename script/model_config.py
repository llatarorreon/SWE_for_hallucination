"""
Model configuration file
Contains configuration information for all supported models and client initialization logic
"""

import os
from typing import Dict, Any, Optional, List, Union
from openai import OpenAI

# Default API configuration
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_API_BASE = (
    os.getenv("OPENAI_API_BASE")
    or os.getenv("OPENAI_BASE_URL")
    or "https://api.openai.com/v1"
)

LOCAL_API_BASE = "http://10.140.54.15:10099/v1/"
LOCAL_API_KEY = None

# Model configuration dictionary
MODEL_CONFIGS = {
    # OpenAI models
    "gpt-4o-2024-11-20": {
        "model_id": "gpt-4o-2024-11-20",
        "temperature": 0.0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "description": "GPT-4o main model",
        "api_key": os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY),
        "base_url": os.getenv("OPENAI_API_BASE", DEFAULT_API_BASE),
    },
    "gpt-4o-mini-2024-07-18": {
        "model_id": "gpt-4o-mini-2024-07-18",
        "temperature": 0.0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "description": "GPT-4o mini version",
        "api_key": os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY),
        "base_url": os.getenv("OPENAI_API_BASE", DEFAULT_API_BASE),
    },
    "claude-3-5-sonnet-20240620": {
        "model_id": "claude-3-5-sonnet-20240620",
        "temperature": 0.0,
        "max_tokens": 4096,
        "supports_system_prompt": True,
        "description": "Claude 3.5 Sonnet (via OpenAI API)",
        "api_key": os.getenv("CLAUDE_API_KEY", DEFAULT_API_KEY),
        "base_url": os.getenv("CLAUDE_API_BASE", DEFAULT_API_BASE),
    },
    "gemini-2.0-flash": {
        "model_id": "gemini-2.0-flash",
        "temperature": 0.0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "description": "Google Gemini 2.0 Flash (via OpenAI API)",
        "api_key": os.getenv("GEMINI_API_KEY", DEFAULT_API_KEY),
        "base_url": os.getenv("GEMINI_API_BASE", DEFAULT_API_BASE),
    },
    "deepseek-v3-0324": {
        "model_id": "deepseek-v3-0324",
        "temperature": 0.0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "description": "DeepSeek v3 (via OpenAI API)",
        "api_key": os.getenv("DEEPSEEK_API_KEY", DEFAULT_API_KEY),
        "base_url": os.getenv("DEEPSEEK_API_BASE", DEFAULT_API_BASE),
    },
    "deepseek-r1": {
        "model_id": "deepseek-reasoner",
        "temperature": 0.0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "description": "DeepSeek R1 (via OpenAI API)",
        "api_key": os.getenv("DEEPSEEK_R1_API_KEY", DEFAULT_API_KEY),
        "base_url": os.getenv("DEEPSEEK_R1_API_BASE", DEFAULT_API_BASE),
    },
    "llama-3.3-70b-instruct": {
        "model_id": "llama-3.3-70b-instruct",
        "temperature": 0.0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "description": "Llama 3.3 70B Instruct (via OpenAI API)",
    },
    "llama-3-70b-instruct": {
        "model_id": "llama-3-70b-instruct",
        "temperature": 0.0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "description": "Llama 3.3 70B Instruct (via OpenAI API)",
    },
    "Qwen2.5-7B-Instruct": {
        "model_id": "Qwen2.5-7B-Instruct",
        "temperature": 0.0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "description": "Qwen 2.5 7B Instruct (local deployment)",
        "api_key": LOCAL_API_KEY,
        "base_url": "http://127.0.0.1:10099/v1/",
    },
    "Qwen2.5-32B-Instruct": {
        "model_id": "Qwen2.5-32B-Instruct",
        "temperature": 0.0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "description": "Qwen 2.5 32B Instruct (local deployment)",
        "api_key": LOCAL_API_KEY,
        "base_url": LOCAL_API_BASE,
    },
    "Qwen2.5-72B-Instruct": {
        "model_id": "Qwen2.5-72B-Instruct",
        "temperature": 0.0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "description": "Qwen 2.5 72B Instruct (local deployment)",
        "api_key": LOCAL_API_KEY,
        "base_url": LOCAL_API_BASE,
    },
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for the specified model

    Args:
        model_name: Model name

    Returns:
        Model configuration dictionary

    Raises:
        ValueError: If the model does not exist
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Supported models: {', '.join(MODEL_CONFIGS.keys())}"
        )

    return MODEL_CONFIGS[model_name]


def get_client(model_name: str) -> OpenAI:
    """
    Get OpenAI client instance for the specified model

    Args:
        model_name: Model name for retrieving the corresponding API configuration

    Returns:
        OpenAI client instance

    Raises:
        EnvironmentError: If necessary API configuration is missing
    """
    # Get model configuration
    model_config = get_model_config(model_name)
    api_key = model_config.get("api_key")
    base_url = model_config.get("base_url")

    client_args = {}
    client_args["base_url"] = base_url.strip()
    print(f"Model {model_name} using API base URL: {base_url}")

    # Create client
    return OpenAI(api_key=api_key, **client_args)


def format_prompt(
    model_name: str,
    user_input: Union[str, List[Dict[str, Any]]],
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Format prompt according to model

    Args:
        model_name: Model name
        user_input: User input content, can be a string or pre-formatted message list (for multimodal input)
        system_prompt: System prompt, some models may not support this

    Returns:
        Formatted prompt dictionary, can be used directly for API calls
    """
    model_config = get_model_config(model_name)
    supports_system_prompt = model_config.get("supports_system_prompt", True)

    # If user input is already a formatted message list (multimodal input), use it directly
    if isinstance(user_input, list):
        # In this case, the input is already a processed message list
        # system_prompt should have been handled in make_prompt and query_model
        return {"messages": user_input}

    # Process normal text input
    if supports_system_prompt and system_prompt:
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
        }
    else:
        # For models that don't support system prompts, combine system prompt and user input
        if system_prompt:
            combined_input = f"{system_prompt}\n\n{user_input}"
        else:
            combined_input = user_input

        return {"messages": [{"role": "user", "content": combined_input}]}


def get_available_models() -> List[str]:
    """
    Get list of all available models

    Returns:
        List of model names
    """
    return list(MODEL_CONFIGS.keys())
