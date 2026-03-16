from typing import Any, Literal

from rlm.environments.base_env import (
    RESERVED_TOOL_NAMES,
    BaseEnv,
    SupportsCustomTools,
    SupportsPersistence,
    ToolInfo,
    extract_tool_value,
    format_tools_for_prompt,
    parse_custom_tools,
    parse_tool_entry,
    validate_custom_tools,
)
from rlm.environments.local_repl import LocalREPL

__all__ = [
    "BaseEnv",
    "LocalREPL",
    "RESERVED_TOOL_NAMES",
    "SupportsCustomTools",
    "SupportsPersistence",
    "ToolInfo",
    "extract_tool_value",
    "format_tools_for_prompt",
    "get_environment",
    "parse_custom_tools",
    "parse_tool_entry",
    "validate_custom_tools",
]


def get_environment(
    environment: Literal["local"],
    environment_kwargs: dict[str, Any],
) -> BaseEnv:
    """
    Routes a specific environment and the args (as a dict) to the appropriate environment if supported.
    Currently supported environments: ['local']
    """
    if environment == "local":
        return LocalREPL(**environment_kwargs)
    else:
        raise ValueError(
            f"Unknown environment: {environment}. Supported: ['local']"
        )


def create_environment(
    environment: Literal["local"],
    environment_kwargs: dict[str, Any],
) -> BaseEnv:
    """Backward-compatible alias for legacy call sites."""
    return get_environment(environment=environment, environment_kwargs=environment_kwargs)
