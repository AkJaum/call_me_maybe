"""Schema-constrained token generation using only the public LLM SDK API."""

import json
import math
from pathlib import Path
import re
from time import perf_counter
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    PrivateAttr,
)

from src.grammar import FunctionCallGrammar, PrefixStatus
from src.model import QwenClient
from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    GeneratedFunctionCall,
    TypeDefinition,
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
    ) -> "ConstrainedDecoder":
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
                parameters,
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
        known_parameters: dict[str, object] | None = None,
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
        source_text = _find_source_text(known_parameters or {})
        pattern_parameter = _is_pattern_parameter(parameter_name, value_type)
        token_limit = self.config.max_new_tokens
        if pattern_parameter and source_text is not None:
            token_limit = min(token_limit, 64)
        for offset in range(token_limit):
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
                normalized = _normalize_generated_value(value, value_type)
                if pattern_parameter and isinstance(normalized, str):
                    if source_text is None or _pattern_matches_source(
                        normalized, source_text
                    ):
                        return normalized, steps
                    return self._select_pattern_fallback(
                        prompt, source_text
                    ), steps
                return _repair_near_verbatim_string(normalized, prompt), steps
        if pattern_parameter and source_text is not None:
            return self._select_pattern_fallback(prompt, source_text), steps
        raise GenerationError(
            f"parameter {parameter_name!r} exceeded "
            f"{self.config.max_new_tokens} tokens; unfinished value suffix: "
            f"{grammar.prefix[len(header):][-120:]!r}"
        )

    def _select_pattern_fallback(self, prompt: str, source_text: str) -> str:
        """Let Qwen select a concise source-matching regex after free-form failure."""
        candidates = _pattern_candidates(prompt, source_text)
        if not candidates:
            raise GenerationError("no source-matching regex fallback is available")
        strategies = [
            FunctionDefinition(
                name=name,
                description=description,
                parameters={},
                returns=TypeDefinition(type="string"),
            )
            for name, _, description in candidates
        ]
        selection_prompt = (
            "Choose the regex strategy matching what the request replaces in the "
            f"source text, never the replacement value. Request: {prompt}"
        )
        selected = self._select_function(selection_prompt, strategies)
        return next(
            pattern for name, pattern, _ in candidates if name == selected.name
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
        prefix = ""
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
    """Build a focused non-thinking chat prompt for one argument value."""
    signature = ", ".join(
        f"{name}:{definition.type}"
        for name, definition in function.parameters.items()
    )
    parameter_guidance = ""
    if value_type == "string":
        normalized_name = parameter_name.casefold()
        parameter_guidance = (
            "When the request supplies a literal string, entity, path, or template "
            "for this parameter, copy only that value verbatim and use JSON escapes. "
            "When it represents an instruction, pattern, format, or mode, derive the "
            "concise machine-readable value required by the function. "
        )
        if "regex" in normalized_name or "pattern" in normalized_name:
            parameter_guidance += (
                "For a regular expression or pattern, use Python syntax without "
                "slash delimiters, match only the requested occurrences, and use "
                "word boundaries when the request targets a complete word. Interpret "
                r"common descriptions using standard syntax: digits as \d+, "
                r"vowels as [aeiouAEIOU], and complete word X as \bX\b. "
            )
        if "replacement" in normalized_name or "substitution" in normalized_name:
            parameter_guidance += (
                "For this replacement parameter, return exactly one corresponding "
                "symbol, such as * for an asterisk, unless a repeated count is "
                "explicitly requested. "
            )
        else:
            parameter_guidance += (
                "For a symbolic replacement described by name, return exactly one "
                "corresponding symbol, such as * for an asterisk, unless a repeated "
                "count is explicitly requested. "
            )
    return (
        "<|im_start|>system\n"
        "Extract exactly one function argument. Return only the concise value "
        "inside the already-started JSON and close the JSON immediately. "
        "Return the value for the named target parameter, never the complete "
        "request or its command words. "
        f"{parameter_guidance}"
        "Respect the target parameter name and never return a sibling parameter's "
        "value. Never explain or show reasoning.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Function: {function.name} - {function.description}\n"
        f"Signature: {function.name}({signature})\n"
        f"Parameter: {parameter_name} ({value_type})\n"
        f"Request: {prompt}\n/no_think<|im_end|>\n"
        "<|im_start|>assistant\nJSON:"
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


def _normalize_generated_value(value: object, value_type: str) -> object:
    """Use an unambiguous Python representation for a declared JSON number."""
    if (
        value_type == "number"
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        return float(value)
    return value


def _is_pattern_parameter(parameter_name: str, value_type: str) -> bool:
    """Recognize string fields whose declared role is a regex or pattern."""
    normalized_name = parameter_name.casefold()
    return value_type == "string" and (
        "regex" in normalized_name or "pattern" in normalized_name
    )


def _find_source_text(parameters: dict[str, object]) -> str | None:
    """Find an already generated source-like string for pattern validation."""
    preferred_terms = ("source", "text", "input")
    for name, value in parameters.items():
        normalized_name = name.casefold()
        if isinstance(value, str) and any(
            term in normalized_name for term in preferred_terms
        ):
            return value
    return None


def _pattern_matches_source(pattern: str, source_text: str) -> bool:
    """Return whether a generated Python regex compiles and matches its source."""
    try:
        return re.search(pattern, source_text) is not None
    except re.error:
        return False


def _pattern_candidates(
    prompt: str,
    source_text: str,
) -> list[tuple[str, str, str]]:
    """Build standard regex strategies and keep only source-matching candidates."""
    candidates: list[tuple[str, str, str]] = [
        ("pattern_digits", r"\d+", "Match one or more source digits or numbers."),
        ("pattern_vowels", "[aeiouAEIOU]", "Match each individual source vowel."),
    ]
    quoted_values = re.findall(r"'([^']+)'|\"([^\"]+)\"", prompt)
    for index, pair in enumerate(quoted_values, start=1):
        value = pair[0] or pair[1]
        if re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
            continue
        safe_name = value.replace("-", "_")[:32]
        candidates.extend(
            [
                (
                    f"pattern_word_{index}_{safe_name}",
                    rf"\b{re.escape(value)}\b",
                    f"Match only the complete source word {value!r}.",
                ),
                (
                    f"pattern_literal_{index}_{safe_name}",
                    re.escape(value),
                    f"Match source text {value!r} literally.",
                ),
            ]
        )
    return [
        candidate
        for candidate in candidates
        if _pattern_matches_source(candidate[1], source_text)
    ]


def _repair_near_verbatim_string(value: object, prompt: str) -> object:
    """Recover a unique prompt substring after at most three copy edits."""
    if not isinstance(value, str) or len(value) < 4 or value in prompt:
        return value

    maximum_edits = 3
    candidates: set[str] = set()
    best_distance = maximum_edits + 1
    minimum_length = max(1, len(value) - maximum_edits)
    maximum_length = min(len(prompt), len(value) + maximum_edits)
    for candidate_length in range(minimum_length, maximum_length + 1):
        for start in range(len(prompt) - candidate_length + 1):
            candidate = prompt[start:start + candidate_length]
            distance = _bounded_edit_distance(
                value,
                candidate,
                maximum_edits,
            )
            if distance < best_distance:
                best_distance = distance
                candidates = {candidate}
            elif distance == best_distance:
                candidates.add(candidate)

    if best_distance <= maximum_edits and len(candidates) == 1:
        return candidates.pop()
    return value


def _bounded_edit_distance(left: str, right: str, limit: int) -> int:
    """Return Levenshtein distance, capped just above a requested limit."""
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return min(previous[-1], limit + 1)
