"""Tests for schema-aware incremental JSON grammar."""

import json

import pytest
from pydantic import ValidationError

from src.grammar import FunctionCallGrammar, PrefixStatus
from src.models import FunctionDefinition, TypeDefinition


def make_function(
    function_name: str = "fn_mix", **parameters: TypeDefinition
) -> FunctionDefinition:
    """Create a function schema while preserving parameter insertion order."""
    return FunctionDefinition(
        name=function_name,
        description="Test grammar.",
        parameters=parameters,
        returns=TypeDefinition(type="string"),
    )


def test_grammar_accepts_complete_scalar_document_incrementally() -> None:
    """Accept a valid document one fragment at a time and finish exactly once."""
    function = make_function(
        text=TypeDefinition(type="string"),
        count=TypeDefinition(type="integer"),
        ratio=TypeDefinition(type="number"),
        enabled=TypeDefinition(type="boolean"),
    )
    document = (
        '{"name":"fn_mix","parameters":{"text":"line\\n\u00e7",'
        '"count":-2,"ratio":1.25e+2,"enabled":false}}'
    )
    grammar = FunctionCallGrammar(functions=(function,))
    for fragment in (
        document[:9],
        document[9:31],
        document[31:57],
        document[57:],
    ):
        assert grammar.can_accept(fragment)
        grammar = grammar.advance(fragment)

    assert grammar.status() is PrefixStatus.COMPLETE
    assert not grammar.can_accept(" ")
    assert json.loads(grammar.prefix)["parameters"]["count"] == -2


def test_grammar_keeps_function_selection_open_for_the_llm() -> None:
    """Allow prefixes for every declared name and reject undeclared names."""
    add = make_function("fn_add", a=TypeDefinition(type="number"))
    greet = make_function("fn_greet", name=TypeDefinition(type="string"))
    grammar = FunctionCallGrammar(functions=(add, greet))

    shared = grammar.advance('{"name":"fn_')
    assert shared.can_accept('add","parameters":{"a":2}}')
    assert shared.can_accept('greet","parameters":{"name":"Ada"}}')
    assert not shared.can_accept('delete","parameters":{}}')


@pytest.mark.parametrize(
    "document",
    [
        '{"name":"fn_int","parameters":{"value":1.5}}',
        '{"name":"fn_int","parameters":{"value":01}}',
        '{"name":"fn_int","parameters":{"value":true}}',
        '{"name":"fn_int","parameters":{"other":1}}',
        '{"parameters":{"value":1},"name":"fn_int"}',
        '{"name":"fn_unknown","parameters":{"value":1}}',
        '{"name":"fn_int","parameters":{"value":1},"extra":0}',
    ],
)
def test_grammar_rejects_schema_or_structure_violations(document: str) -> None:
    """Reject wrong names, keys, types, order, and extra output content."""
    function = make_function("fn_int", value=TypeDefinition(type="integer"))
    grammar = FunctionCallGrammar(functions=(function,))
    assert not grammar.can_accept(document)


def test_grammar_handles_empty_parameters_and_string_escapes() -> None:
    """Support zero-argument functions and valid JSON string escapes."""
    no_args = make_function("fn_ping")
    ping = FunctionCallGrammar(functions=(no_args,)).advance(
        '{"name":"fn_ping","parameters":{}}'
    )
    assert ping.status() is PrefixStatus.COMPLETE

    echo = make_function("fn_echo", text=TypeDefinition(type="string"))
    grammar = FunctionCallGrammar(functions=(echo,))
    assert grammar.can_accept(
        '{"name":"fn_echo","parameters":{"text":"quote: \\""}}'
    )
    assert not grammar.can_accept(
        r'{"name":"fn_echo","parameters":{"text":"bad\x"}}'
    )


def test_invalid_prefix_cannot_become_a_grammar_state() -> None:
    """Protect callers from bypassing advance with an invalid initial prefix."""
    function = make_function("fn_ping")
    with pytest.raises(ValidationError, match="cannot produce"):
        FunctionCallGrammar(functions=(function,), prefix="not-json")


@pytest.mark.parametrize(
    "value_type,value",
    [
        ("number", "-1234567890.25e-3"),
        ("integer", "-999999999999999999"),
        ("boolean", "true"),
        ("boolean", "false"),
        ("string", '"symbols: \\" \\\\ / ç"'),
        ("string", '""'),
    ],
)
def test_grammar_accepts_scalar_edge_cases(
    value_type: str, value: str
) -> None:
    """Accept large, signed, escaped, Unicode, boolean, and empty scalar values."""
    function = make_function("fn_value", value=TypeDefinition(type=value_type))
    document = f'{{"name":"fn_value","parameters":{{"value":{value}}}}}'
    grammar = FunctionCallGrammar(functions=(function,)).advance(document)
    assert grammar.status() is PrefixStatus.COMPLETE
