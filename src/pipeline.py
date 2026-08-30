"""Application pipeline for processing every prompt with one decoder."""

from src.generation import ConstrainedDecoder, GenerationTrace
from src.models import FunctionCallResult, FunctionDefinition, PromptInput


def generate_results(
    decoder: ConstrainedDecoder,
    prompts: list[PromptInput],
    functions: list[FunctionDefinition],
) -> list[FunctionCallResult]:
    """Generate one validated function call per prompt, preserving input
    order."""
    results: list[FunctionCallResult] = []
    for prompt in prompts:
        results.append(decoder.generate(prompt.prompt, functions))
    return results


def generate_results_with_traces(
    decoder: ConstrainedDecoder,
    prompts: list[PromptInput],
    functions: list[FunctionDefinition],
) -> tuple[list[FunctionCallResult], list[GenerationTrace]]:
    """Generate a batch and preserve optional per-token diagnostic traces."""
    traces: list[GenerationTrace] = []
    for prompt in prompts:
        traces.append(decoder.generate_with_trace(prompt.prompt, functions))
    return [trace.result for trace in traces], traces
