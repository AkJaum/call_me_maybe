"""Tests for public Qwen vocabulary loading and grammar filtering."""

import json
from pathlib import Path

import pytest

from src.grammar import FunctionCallGrammar
from src.models import FunctionDefinition, TypeDefinition
from src.vocabulary import TokenVocabulary, VocabularyError


def write_vocabulary(path: Path, vocabulary: dict[str, int]) -> None:
    """Write a minimal byte-level vocabulary fixture."""
    path.write_text(
        json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8"
    )


def test_byte_level_tokens_decode_to_real_text(tmp_path: Path) -> None:
    """Decode Qwen/GPT-2 space markers, UTF-8 bytes, and ASCII syntax."""
    path = tmp_path / "vocab.json"
    write_vocabulary(
        path,
        {
            "{": 0,
            "Ġhello": 1,
            "Ã§": 2,
            "Ã": 3,
        },
    )
    vocabulary = TokenVocabulary.from_file(path)

    assert vocabulary.fragments[0] == "{"
    assert vocabulary.fragments[1] == " hello"
    assert vocabulary.fragments[2] == "ç"
    assert 3 not in vocabulary.fragments
    assert vocabulary.skipped_tokens == 1


def test_vocabulary_filters_whole_fragments_through_grammar(
    tmp_path: Path,
) -> None:
    """Keep only IDs whose complete decoded token preserves the schema."""
    path = tmp_path / "vocab.json"
    write_vocabulary(path, {'{"name":"': 0, "#": 1, "{": 2})
    function = FunctionDefinition(
        name="fn_ping",
        description="Ping.",
        parameters={},
        returns=TypeDefinition(type="boolean"),
    )
    grammar = FunctionCallGrammar(functions=(function,))
    vocabulary = TokenVocabulary.from_file(path)

    assert vocabulary.allowed_token_ids(grammar, logits_count=3) == (0, 2)
    assert vocabulary.allowed_token_ids(grammar, logits_count=1) == (0,)
    assert vocabulary.cache_statistics().model_dump() == {
        "hits": 0,
        "misses": 2,
        "entries": 2,
    }

    assert vocabulary.allowed_token_ids(grammar, logits_count=3) == (0, 2)
    assert vocabulary.cache_statistics().model_dump() == {
        "hits": 1,
        "misses": 2,
        "entries": 2,
    }


def test_invalid_vocabulary_is_reported_cleanly(tmp_path: Path) -> None:
    """Reject malformed roots, entries, and duplicate identifiers."""
    invalid_documents: list[object] = [
        [],
        {"token": "zero"},
        {"a": 0, "b": 0},
        {"negative": -1},
        {},
    ]
    for index, document in enumerate(invalid_documents):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(VocabularyError):
            TokenVocabulary.from_file(path)

    with pytest.raises(VocabularyError, match="not found"):
        TokenVocabulary.from_file(tmp_path / "missing.json")


def test_malformed_vocabulary_json_reports_location(tmp_path: Path) -> None:
    """Expose a readable JSON error instead of leaking the decoder exception."""
    path = tmp_path / "vocab.json"
    path.write_text("{]", encoding="utf-8")
    with pytest.raises(VocabularyError, match="line 1, column 2"):
        TokenVocabulary.from_file(path)


def test_invalid_utf8_vocabulary_is_reported_cleanly(tmp_path: Path) -> None:
    """Translate invalid tokenizer bytes into a vocabulary-domain error."""
    path = tmp_path / "vocab.json"
    path.write_bytes(b'{"\xff": 0}')

    with pytest.raises(VocabularyError, match="not valid UTF-8"):
        TokenVocabulary.from_file(path)
