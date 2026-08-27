"""Validated input, schema, and output data models."""

import math
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationInfo,
    field_validator,
    model_validator,
)


JsonType = Literal["string", "number", "integer", "boolean"]
ScalarValue = str | int | float | bool


class TypeDefinition(BaseModel):
    """Describe one value in a function schema."""

    model_config = ConfigDict(extra="forbid")

    type: JsonType


class FunctionDefinition(BaseModel):
    """Describe a function exposed to the language model."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, TypeDefinition]
    returns: TypeDefinition

    @field_validator("parameters")
    @classmethod
    def validate_parameter_names(
        cls, parameters: dict[str, TypeDefinition]
    ) -> dict[str, TypeDefinition]:
        """Reject empty parameter names, which cannot form a useful schema."""
        if any(not name for name in parameters):
            raise ValueError("parameter names must not be empty")
        return parameters


class FunctionCatalog(RootModel[list[FunctionDefinition]]):
    """Validate the complete collection of callable functions."""

    @model_validator(mode="after")
    def validate_catalog(self) -> "FunctionCatalog":
        """Require at least one function and reject ambiguous duplicate names."""
        if not self.root:
            raise ValueError("at least one function definition is required")
        names = [function.name for function in self.root]
        if len(names) != len(set(names)):
            raise ValueError("function names must be unique")
        return self


class PromptInput(BaseModel):
    """Represent one natural-language function-calling request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class GeneratedFunctionCall(BaseModel):
    """Represent the exact JSON object generated under grammar constraints."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    parameters: dict[str, ScalarValue]


class FunctionCallResult(BaseModel):
    """Represent the exact object written to the output JSON array."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str
    name: str
    parameters: dict[str, ScalarValue]

    @model_validator(mode="after")
    def validate_against_catalog(self, info: ValidationInfo) -> "FunctionCallResult":
        """Validate the selected function and arguments using context definitions."""
        context = info.context
        if not isinstance(context, dict) or "functions" not in context:
            raise ValueError("function definitions are required for result validation")
        functions = context["functions"]
        if not isinstance(functions, list) or not all(
            isinstance(function, FunctionDefinition) for function in functions
        ):
            raise ValueError("invalid function definitions validation context")

        selected = next(
            (function for function in functions if function.name == self.name), None
        )
        if selected is None:
            raise ValueError(f"unknown function: {self.name}")
        expected = set(selected.parameters)
        received = set(self.parameters)
        if expected != received:
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            details: list[str] = []
            if missing:
                details.append(f"missing parameters: {', '.join(missing)}")
            if extra:
                details.append(f"extra parameters: {', '.join(extra)}")
            raise ValueError("; ".join(details))

        for name, definition in selected.parameters.items():
            value = self.parameters[name]
            if not _matches_type(value, definition.type):
                raise ValueError(
                    f"parameter {name!r} must have type {definition.type}"
                )
        return self


def _matches_type(value: ScalarValue, expected: JsonType) -> bool:
    """Return whether a Python scalar exactly matches a JSON schema type."""
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    )


def build_function_call_result(
    prompt: str,
    name: str,
    parameters: dict[str, ScalarValue],
    functions: list[FunctionDefinition],
) -> FunctionCallResult:
    """Build a result only when it exactly satisfies a declared function schema."""
    payload = {"prompt": prompt, "name": name, "parameters": parameters}
    return FunctionCallResult.model_validate(
        payload, strict=True, context={"functions": functions}
    )
