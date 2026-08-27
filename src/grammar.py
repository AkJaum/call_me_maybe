"""Incremental JSON grammar constrained by the available function schemas."""

import json
import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models import FunctionDefinition, JsonType


_JSON_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_NUMBER_PREFIX = re.compile(
    r"(?:|-|-?0|-?[1-9][0-9]*|-?(?:0|[1-9][0-9]*)\."
    r"|-?(?:0|[1-9][0-9]*)\.[0-9]+"
    r"|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?[eE]"
    r"|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?[eE][+-]?"
    r"|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?[eE][+-]?[0-9]+)"
)


class PrefixStatus(str, Enum):
    """Describe whether text is invalid, extendable, or a complete document."""

    INVALID = "invalid"
    PREFIX = "prefix"
    COMPLETE = "complete"


class _MatchResult(BaseModel):
    """Represent an internal grammar match without unvalidated state."""

    model_config = ConfigDict(frozen=True)

    status: PrefixStatus
    position: int = 0


class FunctionCallGrammar(BaseModel):
    """Track a canonical function-call JSON prefix against several schemas.

    The generated document intentionally excludes the original prompt. The caller
    already owns that value and appends it when building ``FunctionCallResult``.
    """

    model_config = ConfigDict(frozen=True)

    functions: tuple[FunctionDefinition, ...] = Field(min_length=1)
    prefix: str = ""

    @model_validator(mode="after")
    def validate_prefix(self) -> "FunctionCallGrammar":
        """Prevent construction of grammar states that cannot be completed."""
        if self.status() is PrefixStatus.INVALID:
            raise ValueError("prefix cannot produce a schema-compliant function call")
        return self

    def status(self) -> PrefixStatus:
        """Return the best match status across all available function schemas."""
        statuses = [
            _match_document(self.prefix, function) for function in self.functions
        ]
        if any(status is PrefixStatus.COMPLETE for status in statuses):
            return PrefixStatus.COMPLETE
        if any(status is PrefixStatus.PREFIX for status in statuses):
            return PrefixStatus.PREFIX
        return PrefixStatus.INVALID

    def can_accept(self, fragment: str) -> bool:
        """Return whether an entire character or token fragment preserves validity."""
        if not fragment or self.status() is PrefixStatus.COMPLETE:
            return False
        candidate = self.prefix + fragment
        return any(
            _match_document(candidate, function) is not PrefixStatus.INVALID
            for function in self.functions
        )

    def advance(self, fragment: str) -> "FunctionCallGrammar":
        """Return a new immutable grammar state after accepting a fragment."""
        if not self.can_accept(fragment):
            raise ValueError(f"invalid constrained-decoding fragment: {fragment!r}")
        return self.model_copy(update={"prefix": self.prefix + fragment})


def _match_document(text: str, function: FunctionDefinition) -> PrefixStatus:
    """Match one canonical document against one concrete function schema."""
    position = 0
    fixed_parts = [
        '{"name":',
        json.dumps(function.name, ensure_ascii=True, separators=(",", ":")),
        ',"parameters":{',
    ]
    for part in fixed_parts:
        match = _match_fixed(text, position, part)
        if match.status is not PrefixStatus.COMPLETE:
            return match.status
        position = match.position

    parameters = list(function.parameters.items())
    for index, (name, definition) in enumerate(parameters):
        if index:
            match = _match_fixed(text, position, ",")
            if match.status is not PrefixStatus.COMPLETE:
                return match.status
            position = match.position
        key = json.dumps(name, ensure_ascii=True, separators=(",", ":")) + ":"
        match = _match_fixed(text, position, key)
        if match.status is not PrefixStatus.COMPLETE:
            return match.status
        position = match.position
        value_match = _match_value(text, position, definition.type)
        if value_match.status is not PrefixStatus.COMPLETE:
            return value_match.status
        position = value_match.position

    closing = _match_fixed(text, position, "}}")
    if closing.status is not PrefixStatus.COMPLETE:
        return closing.status
    if closing.position != len(text):
        return PrefixStatus.INVALID
    return PrefixStatus.COMPLETE


def _match_fixed(text: str, position: int, expected: str) -> _MatchResult:
    """Match fixed syntax, distinguishing an unfinished prefix from a mismatch."""
    available = text[position:position + len(expected)]
    if not expected.startswith(available):
        return _MatchResult(status=PrefixStatus.INVALID)
    if len(available) < len(expected):
        return _MatchResult(status=PrefixStatus.PREFIX)
    return _MatchResult(status=PrefixStatus.COMPLETE, position=position + len(expected))


def _match_value(text: str, position: int, value_type: JsonType) -> _MatchResult:
    """Match one scalar JSON value, leaving structural delimiters unconsumed."""
    if value_type == "string":
        return _match_string(text, position)
    if value_type == "boolean":
        return _match_boolean(text, position)
    return _match_number(text, position, integer_only=value_type == "integer")


def _match_string(text: str, position: int) -> _MatchResult:
    """Match a JSON string, including escapes and Unicode escape sequences."""
    if position == len(text):
        return _MatchResult(status=PrefixStatus.PREFIX)
    if text[position] != '"':
        return _MatchResult(status=PrefixStatus.INVALID)
    index = position + 1
    while index < len(text):
        character = text[index]
        if character == '"':
            return _MatchResult(status=PrefixStatus.COMPLETE, position=index + 1)
        if ord(character) < 0x20:
            return _MatchResult(status=PrefixStatus.INVALID)
        if character != "\\":
            index += 1
            continue
        index += 1
        if index == len(text):
            return _MatchResult(status=PrefixStatus.PREFIX)
        escape = text[index]
        if escape in '"\\/bfnrt':
            index += 1
            continue
        if escape != "u":
            return _MatchResult(status=PrefixStatus.INVALID)
        digits = text[index + 1:index + 5]
        if any(digit not in "0123456789abcdefABCDEF" for digit in digits):
            return _MatchResult(status=PrefixStatus.INVALID)
        if len(digits) < 4:
            return _MatchResult(status=PrefixStatus.PREFIX)
        index += 5
    return _MatchResult(status=PrefixStatus.PREFIX)


def _match_boolean(text: str, position: int) -> _MatchResult:
    """Match either JSON boolean literal."""
    remaining = text[position:]
    candidates = [
        literal for literal in ("true", "false") if literal.startswith(remaining)
    ]
    if candidates:
        if remaining in candidates:
            return _MatchResult(status=PrefixStatus.PREFIX)
        return _MatchResult(status=PrefixStatus.PREFIX)
    for literal in ("true", "false"):
        if remaining.startswith(literal):
            return _MatchResult(
                status=PrefixStatus.COMPLETE,
                position=position + len(literal),
            )
    return _MatchResult(status=PrefixStatus.INVALID)


def _match_number(text: str, position: int, integer_only: bool) -> _MatchResult:
    """Match a finite JSON number or integer without consuming its delimiter."""
    index = position
    while index < len(text) and text[index] in "-+0123456789.eE":
        index += 1
    candidate = text[position:index]
    if index == len(text):
        if _is_number_prefix(candidate, integer_only):
            return _MatchResult(status=PrefixStatus.PREFIX)
        return _MatchResult(status=PrefixStatus.INVALID)
    if not _is_complete_number(candidate, integer_only):
        return _MatchResult(status=PrefixStatus.INVALID)
    return _MatchResult(status=PrefixStatus.COMPLETE, position=index)


def _is_number_prefix(value: str, integer_only: bool) -> bool:
    """Return whether a numeric fragment can become an allowed JSON number."""
    if integer_only:
        return bool(re.fullmatch(r"(?:|-|-?0|-?[1-9][0-9]*)", value))
    return bool(_NUMBER_PREFIX.fullmatch(value))


def _is_complete_number(value: str, integer_only: bool) -> bool:
    """Return whether a numeric fragment is a complete allowed JSON number."""
    if not _JSON_NUMBER.fullmatch(value):
        return False
    return not integer_only or all(character not in value for character in ".eE")
