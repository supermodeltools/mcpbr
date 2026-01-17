"""Model registry for supported LLMs with tool-calling capability."""

from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a supported model."""

    id: str
    provider: str
    display_name: str
    context_window: int
    supports_tools: bool = True
    notes: str = ""


SUPPORTED_MODELS: dict[str, ModelInfo] = {
    # Anthropic models (direct API)
    # Claude 4.5 models
    "claude-opus-4-5-20250514": ModelInfo(
        id="claude-opus-4-5-20250514",
        provider="Anthropic",
        display_name="Claude Opus 4.5",
        context_window=200000,
    ),
    "claude-sonnet-4-5-20250514": ModelInfo(
        id="claude-sonnet-4-5-20250514",
        provider="Anthropic",
        display_name="Claude Sonnet 4.5",
        context_window=200000,
    ),
    "claude-haiku-4-5-20250514": ModelInfo(
        id="claude-haiku-4-5-20250514",
        provider="Anthropic",
        display_name="Claude Haiku 4.5",
        context_window=200000,
    ),
    # Claude 4 models
    "claude-opus-4-20250514": ModelInfo(
        id="claude-opus-4-20250514",
        provider="Anthropic",
        display_name="Claude Opus 4",
        context_window=200000,
    ),
    "claude-sonnet-4-20250514": ModelInfo(
        id="claude-sonnet-4-20250514",
        provider="Anthropic",
        display_name="Claude Sonnet 4",
        context_window=200000,
    ),
    "claude-haiku-4-20250514": ModelInfo(
        id="claude-haiku-4-20250514",
        provider="Anthropic",
        display_name="Claude Haiku 4",
        context_window=200000,
    ),
    # Claude 3.5 models
    "claude-3-5-sonnet-20241022": ModelInfo(
        id="claude-3-5-sonnet-20241022",
        provider="Anthropic",
        display_name="Claude 3.5 Sonnet",
        context_window=200000,
    ),
}

DEFAULT_MODEL = "claude-sonnet-4-5-20250514"


def is_model_supported(model_id: str) -> bool:
    """Check if a model is in the supported list.

    Args:
        model_id: Anthropic model ID.

    Returns:
        True if the model is supported.
    """
    return model_id in SUPPORTED_MODELS


def get_model_info(model_id: str) -> ModelInfo | None:
    """Get information about a model.

    Args:
        model_id: Anthropic model ID.

    Returns:
        ModelInfo if found, None otherwise.
    """
    return SUPPORTED_MODELS.get(model_id)


def list_supported_models() -> list[ModelInfo]:
    """Get a list of all supported models.

    Returns:
        List of ModelInfo objects.
    """
    return list(SUPPORTED_MODELS.values())


def get_models_by_provider(provider: str) -> list[ModelInfo]:
    """Get models filtered by provider.

    Args:
        provider: Provider name (e.g., "Anthropic").

    Returns:
        List of ModelInfo objects from that provider.
    """
    result = []
    for model in SUPPORTED_MODELS.values():
        if model.provider.lower() == provider.lower():
            result.append(model)
    return result


def validate_model(model_id: str) -> tuple[bool, str]:
    """Validate a model ID and return a helpful error message if invalid.

    Args:
        model_id: Anthropic model ID to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if is_model_supported(model_id):
        return True, ""

    supported_ids = list(SUPPORTED_MODELS.keys())
    suggestion = ""

    model_lower = model_id.lower()
    for supported_id in supported_ids:
        if model_lower in supported_id.lower() or supported_id.lower() in model_lower:
            suggestion = f" Did you mean '{supported_id}'?"
            break

    return False, (
        f"Model '{model_id}' is not in the list of supported tool-capable models.{suggestion}\n"
        f"Run 'mcpbr models' to see available models."
    )
