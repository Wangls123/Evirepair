from __future__ import annotations

import os

Y_LLM_API_KEY: str = ""

Y_LLM_BASE_URL: str = ""
Y_LLM_MODEL: str = "qwen-plus"
Y_LLM_TEMPERATURE: float = 0.2
Y_LLM_TOP_P: float = 0.95
Y_LLM_MAX_TOKENS: int = 4096
Y_LLM_TIMEOUT_SEC: int = 600
Y_LLM_SAMPLE_MAX_ATTEMPTS: int = 5
Y_LLM_RETRY_SLEEP_SEC: float = 5.0
Y_LLM_USE_JSON_RESPONSE_FORMAT: bool = True

def resolve_y_llm_api_key() -> str:

    for name in ("DASHSCOPE_API_KEY", "V4_LLM_API_KEY", "LLM_API_KEY"):
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    return str(Y_LLM_API_KEY or "").strip()

def resolve_y_llm_base_url() -> str:
    for name in ("DASHSCOPE_BASE_URL", "Y_LLM_BASE_URL"):
        value = str(os.getenv(name) or "").strip()
        if value:
            return value.rstrip("/")
    return str(Y_LLM_BASE_URL or "").strip().rstrip("/")

def y_llm_configured() -> bool:
    return bool(resolve_y_llm_api_key() and resolve_y_llm_base_url() and Y_LLM_MODEL.strip())

def get_y_llm_config() -> dict:
    return {
        "api_key": resolve_y_llm_api_key(),
        "base_url": resolve_y_llm_base_url(),
        "model": Y_LLM_MODEL.strip(),
        "temperature": Y_LLM_TEMPERATURE,
        "top_p": Y_LLM_TOP_P,
        "max_tokens": Y_LLM_MAX_TOKENS,
        "timeout_sec": Y_LLM_TIMEOUT_SEC,
        "use_json_response_format": Y_LLM_USE_JSON_RESPONSE_FORMAT,
        "sample_max_attempts": Y_LLM_SAMPLE_MAX_ATTEMPTS,
        "retry_sleep_sec": Y_LLM_RETRY_SLEEP_SEC,
    }
