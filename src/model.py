"""Public llm_sdk adapter for the required Qwen model."""

from importlib import import_module
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, PrivateAttr

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


class ModelLoadError(RuntimeError):
    """Report a model or SDK initialization failure."""


class QwenClient(BaseModel):
    """Expose only the public SDK operations needed by constrained decoding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = DEFAULT_MODEL
    _sdk: Any = PrivateAttr(default=None)

    def load(self) -> None:
        """Load the SDK lazily so input validation does not download a
        model."""
        try:
            sdk_class: Any = getattr(
                import_module("llm_sdk"), "Small_LLM_Model"
            )
            self._sdk = sdk_class(model_name=self.model_name)
        except Exception as exc:
            raise ModelLoadError(
                f"could not load {self.model_name}: {exc}"
            ) from exc

    def encode(self, text: str) -> list[int]:
        """Encode text and return a plain list of token identifiers."""
        if self._sdk is None:
            raise ModelLoadError("model is not loaded")
        encoded = self._sdk.encode(text)
        values: list[int] = encoded[0].tolist()
        return values

    def next_token_logits(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits through the SDK public API."""
        if self._sdk is None:
            raise ModelLoadError("model is not loaded")
        return cast(
            list[float], self._sdk.get_logits_from_input_ids(input_ids)
        )

    def vocab_path(self) -> str:
        """Return the SDK-provided vocabulary path."""
        if self._sdk is None:
            raise ModelLoadError("model is not loaded")
        path: str = self._sdk.get_path_to_vocab_file()
        return path
