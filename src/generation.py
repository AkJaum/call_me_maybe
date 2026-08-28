"""Schema-constrained token generation using only the public LLM SDK API."""

import json
import math
from pathlib import Path
from time import perf_counter
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    PrivateAttr,
    ValidationError,
)

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
    _selection_baselines: dict[tuple[str, str], list[float]] = PrivateAttr(
        default_factory=dict
    )

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
        selected = self._select_function(prompt, functions)
        parameters: dict[str, object] = {}
        steps: list[GenerationStep] = []
        step_index = 1
        for parameter_name, definition in selected.parameters.items():
            value, parameter_steps = self._generate_parameter(
                prompt,
                selected,
                parameter_name,
                definition.type,
                step_index,
                collect_trace,
            )
            parameters[parameter_name] = value
            steps.extend(parameter_steps)
            step_index += len(parameter_steps)
        generated = GeneratedFunctionCall.model_validate(
            {"name": selected.name, "parameters": parameters},
            strict=True,
        )
        result = build_function_call_result(
            prompt,
            generated.name,
            generated.parameters,
            functions,
        )
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

    def _generate_parameter(
        self,
        prompt: str,
        function: FunctionDefinition,
        parameter_name: str,
        value_type: str,
        first_step_index: int,
        collect_trace: bool,
    ) -> tuple[object, list[GenerationStep]]:
        """Generate one named argument under its complete one-field schema."""
        definition = function.parameters[parameter_name]
        single_parameter_function = function.model_copy(
            update={"parameters": {parameter_name: definition}}
        )
        header = (
            '{"name":'
            + json.dumps(function.name, ensure_ascii=True, separators=(",", ":"))
            + ',"parameters":{'
            + json.dumps(parameter_name, ensure_ascii=True, separators=(",", ":"))
            + ":"
        )
        if value_type == "string":
            header += '"'
        grammar = FunctionCallGrammar(
            functions=(single_parameter_function,)
        ).advance(header)
        model_prompt = build_parameter_prompt(
            prompt,
            function,
            parameter_name,
            value_type,
        ) + header
        prompt_ids = self.client.encode(model_prompt)
        if not prompt_ids:
            raise GenerationError("the model encoded the prompt as an empty sequence")

        generated_ids: list[int] = []
        steps: list[GenerationStep] = []
        for offset in range(self.config.max_new_tokens):
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
                        index=first_step_index + offset,
                        token_id=token_id,
                        token_fragment=fragment,
                        allowed_token_count=len(allowed_ids),
                        selected_logit=masked_logits[token_id],
                        prefix=grammar.prefix,
                    )
                )
            if grammar.status() is PrefixStatus.COMPLETE:
                try:
                    raw = json.loads(grammar.prefix)
                    value = raw["parameters"][parameter_name]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise GenerationError(
                        "generated parameter failed final JSON validation"
                    ) from exc
                return _normalize_generated_value(value, value_type), steps
        raise GenerationError(
            f"parameter {parameter_name!r} exceeded "
            f"{self.config.max_new_tokens} tokens"
        )

    def _select_function(
        self,
        prompt: str,
        functions: list[FunctionDefinition],
    ) -> FunctionDefinition:
        """Choose a catalog function using context-calibrated constrained logits."""
        if len(functions) == 1:
            return functions[0]
        targets = {function.name: function.name + "\n" for function in functions}
        prefix = _longest_common_prefix(list(targets.values()))
        schema_key = "|".join(function.model_dump_json() for function in functions)
        for _ in range(self.config.max_new_tokens):
            viable = {
                name: target
                for name, target in targets.items()
                if target.startswith(prefix)
            }
            if len(viable) == 1:
                selected_name = next(iter(viable))
                return next(
                    function
                    for function in functions
                    if function.name == selected_name
                )

            actual_prompt = build_selection_prompt(prompt, functions, prefix)
            actual_logits = self.client.next_token_logits(
                self.client.encode(actual_prompt)
            )
            baseline_key = (schema_key, prefix)
            baseline_logits = self._selection_baselines.get(baseline_key)
            if baseline_logits is None:
                baseline_prompt = build_selection_prompt(
                    "[unspecified]",
                    functions,
                    prefix,
                )
                baseline_logits = self.client.next_token_logits(
                    self.client.encode(baseline_prompt)
                )
                self._selection_baselines[baseline_key] = baseline_logits
            if len(actual_logits) != len(baseline_logits):
                raise GenerationError("selection logits changed vocabulary size")

            candidates = [
                token_id
                for token_id, fragment in self.vocabulary.fragments.items()
                if token_id < len(actual_logits)
                and any(
                    target.startswith(prefix + fragment)
                    for target in viable.values()
                )
            ]
            if not candidates:
                raise GenerationError("no token can continue function selection")
            token_id = max(
                candidates,
                key=lambda candidate: (
                    actual_logits[candidate] - baseline_logits[candidate]
                ),
            )
            prefix += self.vocabulary.fragments[token_id]
        raise GenerationError("function selection exceeded the token limit")


def build_parameter_prompt(
    prompt: str,
    function: FunctionDefinition,
    parameter_name: str,
    value_type: str,
) -> str:
    """Build a focused prompt that exposes one parameter name and type."""
    return (
        "Extract exactly the requested parameter from the request. Copy every "
        "string character verbatim, using JSON escapes for embedded quotes. "
        "Stop immediately after the last character of the request. "
        "Do not rewrite or normalize. Return only JSON.\n"
        f"Function: {function.name} - {function.description}\n"
        f"Parameter: {parameter_name} ({value_type})\n"
        f"Request: {prompt}\n"
        "JSON:"
    )


def build_selection_prompt(
    request: str,
    functions: list[FunctionDefinition],
    prefix: str,
) -> str:
    """Build the calibrated function-name classification context."""
    catalog = "\n".join(
        f"- {function.name}({', '.join(function.parameters)}): "
        f"{function.description}"
        for function in functions
    )
    return (
        "<|im_start|>system\n"
        "Select the single function that matches the user's request. "
        "Return only its exact name.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Available functions:\n{catalog}\n"
        f"Request: {request}\n/no_think<|im_end|>\n"
        f"<|im_start|>assistant\n{prefix}"
    )


def _longest_common_prefix(values: list[str]) -> str:
    """Return text shared from the beginning of every non-empty value."""
    if not values:
        return ""
    common = values[0]
    for value in values[1:]:
        limit = min(len(common), len(value))
        index = 0
        while index < limit and common[index] == value[index]:
            index += 1
        common = common[:index]
    return common


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
    """Expand compact generated JSON and apply final dynamic validation."""
    try:
        raw: object = json.loads(document)
        if not isinstance(raw, list) or not raw:
            raise ValueError("generated compact call must be a non-empty array")
        function_name = raw[0]
        if not isinstance(function_name, str):
            raise ValueError("generated function name must be a string")
        selected = next(
            (function for function in functions if function.name == function_name),
            None,
        )
        if selected is None:
            raise ValueError("generated function name is outside the catalog")
        parameter_names = list(selected.parameters)
        if len(raw) != len(parameter_names) + 1:
            raise ValueError("generated compact call has the wrong argument count")
        parameters = {
            name: _normalize_generated_value(raw[index + 1], definition.type)
            for index, (name, definition) in enumerate(selected.parameters.items())
        }
        generated = GeneratedFunctionCall.model_validate(
            {"name": selected.name, "parameters": parameters},
            strict=True,
        )
        return build_function_call_result(
            prompt,
            generated.name,
            generated.parameters,
            functions,
        )
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise GenerationError(f"generated call failed final validation: {exc}") from exc


def _normalize_generated_value(value: object, value_type: str) -> object:
    """Use an unambiguous Python representation for a declared JSON number."""
    if (
        value_type == "number"
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        return float(value)
    return value
