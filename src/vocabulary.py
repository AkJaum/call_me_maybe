"""Load Qwen's public vocabulary and expose decoded token fragments."""

import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.grammar import FunctionCallGrammar


class VocabularyError(ValueError):
    """Report an invalid or unreadable tokenizer vocabulary."""


class CacheStatistics(BaseModel):
    """Expose safe counters for the constrained-token cache."""

    model_config = ConfigDict(frozen=True)

    hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    entries: int = Field(ge=0)


class TokenVocabulary(BaseModel):
    """Map model token identifiers to independently valid UTF-8 fragments."""

    model_config = ConfigDict(frozen=True)

    fragments: dict[int, str] = Field(min_length=1)
    skipped_tokens: int = Field(default=0, ge=0)
    _allowed_cache: dict[tuple[str, str, int], tuple[int, ...]] = PrivateAttr(
        default_factory=dict
    )
    _cache_hits: int = PrivateAttr(default=0)
    _cache_misses: int = PrivateAttr(default=0)
    _first_character_index: dict[str, tuple[int, ...]] = PrivateAttr(
        default_factory=dict
    )
    _plain_string_ids: tuple[int, ...] = PrivateAttr(default=())
    _special_string_ids: tuple[int, ...] = PrivateAttr(default=())

    @model_validator(mode="after")
    def validate_fragments(self) -> "TokenVocabulary":
        """Reject negative IDs and empty fragments at the model boundary."""
        if any(token_id < 0 for token_id in self.fragments):
            raise ValueError("token identifiers must not be negative")
        if any(not fragment for fragment in self.fragments.values()):
            raise ValueError("token fragments must not be empty")
        return self

    def model_post_init(self, __context: object) -> None:
        """Index token categories once instead of scanning the full
        vocabulary."""
        first_character_index: dict[str, list[int]] = {}
        plain_string_ids: list[int] = []
        special_string_ids: list[int] = []
        for token_id, fragment in self.fragments.items():
            first_character_index.setdefault(fragment[0], []).append(token_id)
            if _is_plain_string_fragment(fragment):
                plain_string_ids.append(token_id)
            else:
                special_string_ids.append(token_id)
        self._first_character_index = {
            character: tuple(token_ids)
            for character, token_ids in first_character_index.items()
        }
        self._plain_string_ids = tuple(plain_string_ids)
        self._special_string_ids = tuple(special_string_ids)

    @classmethod
    def from_file(cls, path: Path) -> "TokenVocabulary":
        """Load a Hugging Face byte-level BPE ``vocab.json`` file."""
        try:
            with path.open(encoding="utf-8") as stream:
                raw: object = json.load(stream)
        except FileNotFoundError as exc:
            raise VocabularyError(
                f"vocabulary file not found: {path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise VocabularyError(
                f"invalid vocabulary JSON at line {exc.lineno}, "
                f"column {exc.colno}"
            ) from exc
        except UnicodeDecodeError as exc:
            raise VocabularyError(
                f"vocabulary is not valid UTF-8: {path} at byte {exc.start}"
            ) from exc
        except OSError as exc:
            raise VocabularyError(
                f"could not read vocabulary {path}: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise VocabularyError("vocabulary root must be a JSON object")

        byte_decoder = _byte_decoder()
        fragments: dict[int, str] = {}
        seen_ids: set[int] = set()
        skipped = 0
        for encoded_token, token_id in raw.items():
            if (
                not isinstance(encoded_token, str)
                or not isinstance(token_id, int)
                or isinstance(token_id, bool)
            ):
                raise VocabularyError(
                    "vocabulary must map strings to integer IDs"
                )
            if token_id < 0:
                raise VocabularyError("token identifiers must not be negative")
            if token_id in seen_ids:
                raise VocabularyError(
                    f"duplicate token identifier: {token_id}"
                )
            seen_ids.add(token_id)
            fragment = _decode_token(encoded_token, byte_decoder)
            if fragment is None or not fragment:
                skipped += 1
                continue
            fragments[token_id] = fragment
        if not fragments:
            raise VocabularyError("vocabulary contains no usable UTF-8 tokens")
        return cls(fragments=fragments, skipped_tokens=skipped)

    def allowed_token_ids(
        self, grammar: FunctionCallGrammar, logits_count: int
    ) -> tuple[int, ...]:
        """Return token IDs whose complete fragments preserve the grammar."""
        if logits_count <= 0:
            return ()
        schema_key = "|".join(
            function.model_dump_json() for function in grammar.functions
        )
        cache_key = (
            grammar.__class__.__name__ + ":" + schema_key,
            grammar.prefix,
            logits_count,
        )
        cached = self._allowed_cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        if grammar.is_inside_plain_string():
            plain = (
                token_id
                for token_id in self._plain_string_ids
                if token_id < logits_count
            )
            special = (
                token_id
                for token_id in self._special_string_ids
                if token_id < logits_count
                and grammar.can_accept(self.fragments[token_id])
            )
            allowed = tuple(plain) + tuple(special)
        else:
            valid_initials = {
                character
                for character in self._first_character_index
                if grammar.can_accept(character)
            }
            candidates = (
                token_id
                for character in valid_initials
                for token_id in self._first_character_index[character]
            )
            allowed = tuple(
                token_id
                for token_id in candidates
                if token_id < logits_count
                and grammar.can_accept(self.fragments[token_id])
            )
        self._allowed_cache[cache_key] = allowed
        return allowed

    def cache_statistics(self) -> CacheStatistics:
        """Return an immutable snapshot of cache use for diagnostics."""
        return CacheStatistics(
            hits=self._cache_hits,
            misses=self._cache_misses,
            entries=len(self._allowed_cache),
        )


def _byte_decoder() -> dict[str, int]:
    """Build the inverse GPT-2/Qwen byte-to-Unicode table."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = byte_values.copy()
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return {
        chr(unicode_value): byte_value
        for byte_value, unicode_value in zip(byte_values, unicode_values)
    }


def _decode_token(encoded_token: str, decoder: dict[str, int]) -> str | None:
    """Decode one byte-level BPE token, skipping incomplete UTF-8 sequences."""
    try:
        raw_bytes = bytes(decoder[character] for character in encoded_token)
        return raw_bytes.decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None


def _is_plain_string_fragment(fragment: str) -> bool:
    """Recognize text that cannot close, escape, or invalidate a JSON
    string."""
    return all(
        character not in {'"', "\\"} and ord(character) >= 0x20
        for character in fragment
    )
