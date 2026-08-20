"""Validated input and output data models."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


JsonType = Literal["string", "number", "integer", "boolean", "array", "object"]


class TypeDefinition(BaseModel):
    """Describe one value in a function schema."""

    model_config = ConfigDict(extra="allow")

    type: JsonType


class FunctionDefinition(BaseModel):
    """Describe a function exposed to the language model."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, TypeDefinition]
    returns: TypeDefinition


class PromptInput(BaseModel):
    """Represent one natural-language function-calling request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class FunctionCallResult(BaseModel):
    """Represent the exact object written to the output JSON array."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
