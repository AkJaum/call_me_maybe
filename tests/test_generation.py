"""Tests for logits masking and end-to-end constrained generation."""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.generation import (
    ConstrainedDecoder,
    GenerationConfig,
    GenerationError,
    mask_invalid_logits,
    parse_generated_result,
    select_highest_logit,
)
from src.model import QwenClient
from src.models import FunctionDefinition, TypeDefinition


def make_client(
    path: Path, planned_ids: list[int], vocab_size: int
) -> QwenClient:
    """Attach controlled public SDK operations behind the Qwen adapter."""
    sdk = MagicMock()
    vocabulary: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))

    def make_encoding(token_ids: list[int]) -> MagicMock:
        """Build the two-dimensional tensor shape exposed by the SDK."""
        encoded = MagicMock()
        encoded[0].tolist.return_value = token_ids
        return encoded

    def encode(text: str) -> MagicMock:
        """Encode prompts stably and forced fixture fragments greedily."""
        if "Request:" in text and "JSON:" in text:
            return make_encoding([99])
        position = 0
        token_ids: list[int] = []
        while position < len(text):
            matches = [
                (fragment, token_id)
                for fragment, token_id in vocabulary.items()
                if text.startswith(fragment, position)
            ]
            if not matches:
                return make_encoding([])
            fragment, token_id = max(matches, key=lambda item: len(item[0]))
            token_ids.append(token_id)
            position += len(fragment)
        return make_encoding(token_ids)

    logits_call = 0

    def get_logits(input_ids: list[int]) -> list[float]:
        """Prefer an invalid token globally and the planned valid token second."""
        nonlocal logits_call
        logits = [-10.0] * vocab_size
        logits[vocabulary["#"]] = 100.0
        logits[planned_ids[logits_call % len(planned_ids)]] = 10.0
        logits_call += 1
        return logits

    sdk.encode.side_effect = encode
    sdk.get_logits_from_input_ids.side_effect = get_logits
    sdk.get_path_to_vocab_file.return_value = str(path)
    client = QwenClient()
    client._sdk = sdk
    return client


def test_mask_sets_every_invalid_logit_to_negative_infinity() -> None:
    """Apply the constrained-decoding operation required by the subject."""
    masked = mask_invalid_logits([1.0, 5.0, math.nan], (0, 2))
    assert masked == [1.0, -math.inf, -math.inf]
    assert select_highest_logit(masked) == 0

    with pytest.raises(GenerationError, match="no vocabulary token"):
        select_highest_logit([-math.inf])

    with pytest.raises(GenerationError, match="empty logits"):
        select_highest_logit([])


def test_decoder_uses_model_logits_but_blocks_invalid_choice(tmp_path: Path) -> None:
    """Generate a typed call while grammar-masking a higher invalid logit."""
    fragments = ["2", "}}", "3", "#"]
    path = tmp_path / "vocab.json"
    path.write_text(
        json.dumps({fragment: index for index, fragment in enumerate(fragments)}),
        encoding="utf-8",
    )
    client = make_client(path, planned_ids=[0, 1, 2, 1], vocab_size=len(fragments))
    decoder = ConstrainedDecoder.from_client(
        client,
        GenerationConfig(max_new_tokens=16),
    )
    function = FunctionDefinition(
        name="fn_add",
        description="Add two numbers.",
        parameters={
            "a": TypeDefinition(type="number"),
            "b": TypeDefinition(type="number"),
        },
        returns=TypeDefinition(type="number"),
    )

    result = decoder.generate("Add 2 and 3", [function])

    assert result.prompt == "Add 2 and 3"
    assert result.name == "fn_add"
    assert result.parameters == {"a": 2.0, "b": 3.0}

    trace = decoder.generate_with_trace("Add 2 and 3", [function])
    assert trace.result == result
    assert [step.index for step in trace.steps] == list(range(1, 5))
    assert all(step.allowed_token_count >= 1 for step in trace.steps)
    assert trace.steps[-1].prefix == (
        '{"name":"fn_add","parameters":{"b":3}}'
    )
    assert trace.cache_hits == 4
    assert trace.cache_misses == 0


def test_generation_stops_at_configured_token_limit(tmp_path: Path) -> None:
    """Fail gracefully instead of entering an unbounded generation loop."""
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"x": 0, "#": 1}), encoding="utf-8")
    decoder = ConstrainedDecoder.from_client(
        make_client(path, planned_ids=[0], vocab_size=2),
        GenerationConfig(max_new_tokens=1),
    )
    function = FunctionDefinition(
        name="fn_ping",
        description="Ping.",
        parameters={"text": TypeDefinition(type="string")},
        returns=TypeDefinition(type="boolean"),
    )

    with pytest.raises(GenerationError, match="exceeded 1"):
        decoder.generate("Ping", [function])


def test_generation_reports_when_no_token_can_continue(tmp_path: Path) -> None:
    """Return a controlled error when the vocabulary cannot start the grammar."""
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"#": 0}), encoding="utf-8")
    decoder = ConstrainedDecoder.from_client(
        make_client(path, planned_ids=[0], vocab_size=1)
    )
    function = FunctionDefinition(
        name="fn_ping",
        description="Ping.",
        parameters={"value": TypeDefinition(type="integer")},
        returns=TypeDefinition(type="boolean"),
    )

    with pytest.raises(GenerationError, match="no vocabulary token"):
        decoder.generate("Ping", [function])


@pytest.mark.parametrize(
    "document",
    [
        '{"name":"fn_ping","parameters":{},"extra":1}',
        '{"name":"fn_missing","parameters":{}}',
        '{"name":"fn_ping","parameters":{"extra":1}}',
        "not-json",
    ],
)
def test_final_generated_document_is_validated_again(document: str) -> None:
    """Keep final Pydantic validation as defense in depth after the grammar."""
    function = FunctionDefinition(
        name="fn_ping",
        description="Ping.",
        parameters={},
        returns=TypeDefinition(type="boolean"),
    )
    with pytest.raises(GenerationError, match="failed final validation"):
        parse_generated_result(document, "Ping", [function])


def test_compact_number_is_normalized_for_strict_python_consumers() -> None:
    """Write JSON-schema numbers as floats even when Qwen emits an integer."""
    function = FunctionDefinition(
        name="fn_scale",
        description="Scale a number.",
        parameters={"value": TypeDefinition(type="number")},
        returns=TypeDefinition(type="number"),
    )

    result = parse_generated_result('["fn_scale",3]', "Scale 3", [function])

    assert result.parameters == {"value": 3.0}
    assert isinstance(result.parameters["value"], float)
