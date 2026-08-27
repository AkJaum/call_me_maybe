"""Schema-constrained token generation using only the public LLM SDK API."""

import json
import math
from pathlib import Path
from time import perf_counter
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, InstanceOf, ValidationError

from src.grammar import FunctionCallGrammar, PrefixStatus
from src.model import QwenClient
from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    GeneratedFunctionCall,
    build_function_call_result,
)
from src.vocabulary import TokenVocabulary


class GenerationError(RuntimeError):
    """Report a controlled failure in constrained generation."""


class GenerationConfig(BaseModel):
    """Configure bounded deterministic generation."""

    model_config = ConfigDict(frozen=True)

    max_new_tokens: int = Field(default=256, ge=1, le=4096)


class GenerationStep(BaseModel):
    """Record one constrained token decision for optional visualization."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=1)
    token_id: int = Field(ge=0)
    token_fragment: str = Field(min_length=1)
    allowed_token_count: int = Field(ge=1)
    selected_logit: float
    prefix: str = Field(min_length=1)


class GenerationTrace(BaseModel):
    """Describe a complete generation without changing its required result."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    duration_seconds: float = Field(ge=0.0)
    cache_hits: int = Field(ge=0)
    cache_misses: int = Field(ge=0)
    steps: tuple[GenerationStep, ...]
    result: InstanceOf[FunctionCallResult]


class ConstrainedDecoder(BaseModel):
    """Generate function calls by masking every grammar-invalid token."""

    model_config = ConfigDict(frozen=True)

    client: QwenClient
    vocabulary: TokenVocabulary
    config: GenerationConfig = Field(default_factory=GenerationConfig)

    @classmethod
    def from_client(
        cls,
        client: QwenClient,
        config: GenerationConfig | None = None,
    ) -> Self:
        """Load the vocabulary through the SDK's public path operation."""
        vocabulary = TokenVocabulary.from_file(Path(client.vocab_path()))
        return cls(
            client=client,
            vocabulary=vocabulary,
            config=config or GenerationConfig(),
        )

    def generate(
        self,
        prompt: str,
        functions: list[FunctionDefinition],
    ) -> FunctionCallResult:
        """Generate and validate one function call for a natural-language prompt."""
        return self._generate(prompt, functions, collect_trace=False).result

    def generate_with_trace(
        self,
        prompt: str,
        functions: list[FunctionDefinition],
    ) -> GenerationTrace:
        """Generate one call and retain each valid-token decision for display."""
        return self._generate(prompt, functions, collect_trace=True)

    def _generate(
        self,
        prompt: str,
        functions: list[FunctionDefinition],
        collect_trace: bool,
    ) -> GenerationTrace:
        """Run the shared decoder loop with optional trace collection."""
        started_at = perf_counter()
        cache_before = self.vocabulary.cache_statistics()
        model_prompt = build_model_prompt(prompt, functions)
        prompt_ids = self.client.encode(model_prompt)
        if not prompt_ids:
            raise GenerationError("the model encoded the prompt as an empty sequence")

        grammar = FunctionCallGrammar(functions=tuple(functions))
        generated_ids: list[int] = []
        steps: list[GenerationStep] = []
        for index in range(1, self.config.max_new_tokens + 1):
            logits = self.client.next_token_logits(prompt_ids + generated_ids)
            allowed_ids = self.vocabulary.allowed_token_ids(grammar, len(logits))
            masked_logits = mask_invalid_logits(logits, allowed_ids)
            token_id = select_highest_logit(masked_logits)
            fragment = self.vocabulary.fragments[token_id]
            grammar = grammar.advance(fragment)
            generated_ids.append(token_id)
            if collect_trace:
                steps.append(
                    GenerationStep(
                        index=index,
                        token_id=token_id,
                        token_fragment=fragment,
                        allowed_token_count=len(allowed_ids),
                        selected_logit=masked_logits[token_id],
                        prefix=grammar.prefix,
                    )
                )
            if grammar.status() is PrefixStatus.COMPLETE:
                result = parse_generated_result(grammar.prefix, prompt, functions)
                cache_after = self.vocabulary.cache_statistics()
                return GenerationTrace(
                    prompt=prompt,
                    model_name=self.client.model_name,
                    duration_seconds=perf_counter() - started_at,
                    cache_hits=cache_after.hits - cache_before.hits,
                    cache_misses=cache_after.misses - cache_before.misses,
                    steps=tuple(steps),
                    result=result,
                )

        raise GenerationError(
            f"generation exceeded {self.config.max_new_tokens} tokens"
        )


def build_model_prompt(
    prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Describe the task and available functions without hardcoded selection rules."""
    definitions = [
        function.model_dump(mode="json", exclude={"returns"})
        for function in functions
    ]
    serialized = json.dumps(definitions, ensure_ascii=True, separators=(",", ":"))
    return (
        "Choose the matching function and extract its arguments.\n"
        f"Functions: {serialized}\n"
        f"Request: {prompt}\n"
        "JSON:"
    )


def mask_invalid_logits(
    logits: list[float], allowed_ids: tuple[int, ...]
) -> list[float]:
    """Set every invalid or non-finite logit to negative infinity."""
    masked = [-math.inf] * len(logits)
    for token_id in allowed_ids:
        if 0 <= token_id < len(logits) and math.isfinite(logits[token_id]):
            masked[token_id] = logits[token_id]
    return masked


def select_highest_logit(masked_logits: list[float]) -> int:
    """Select the model's highest-scoring token after grammar masking."""
    if not masked_logits:
        raise GenerationError("the model returned an empty logits vector")
    token_id = max(range(len(masked_logits)), key=masked_logits.__getitem__)
    if masked_logits[token_id] == -math.inf:
        raise GenerationError("no vocabulary token can continue the JSON grammar")
    return token_id


def parse_generated_result(
    document: str,
    prompt: str,
    functions: list[FunctionDefinition],
) -> FunctionCallResult:
    """Parse a complete grammar document and apply final dynamic validation."""
    try:
        raw: object = json.loads(document)
        generated = GeneratedFunctionCall.model_validate(raw, strict=True)
        return build_function_call_result(
            prompt,
            generated.name,
            generated.parameters,
            functions,
        )
    except (json.JSONDecodeError, ValidationError) as exc:
        raise GenerationError(f"generated call failed final validation: {exc}") from exc
