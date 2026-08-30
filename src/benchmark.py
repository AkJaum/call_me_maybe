"""Reproducible accuracy, validity, time, and memory benchmark."""

import argparse
import json
from pathlib import Path
import resource
import sys
import time
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from src.generation import ConstrainedDecoder, GenerationError
from src.io import InputFileError, load_function_definitions
from src.model import ModelLoadError, QwenClient
from src.models import FunctionCallResult, ScalarValue
from src.vocabulary import VocabularyError

DEFAULT_CASES = Path("benchmarks/cases.json")
DEFAULT_DEFINITIONS = Path("data/input/functions_definition.json")
DEFAULT_REPORT = Path("benchmarks/latest_results.json")


class BenchmarkCase(BaseModel):
    """Describe one labeled function-calling benchmark case."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(min_length=1)
    expected_name: str = Field(min_length=1)
    expected_parameters: dict[str, ScalarValue]


class BenchmarkReport(BaseModel):
    """Store reproducible aggregate benchmark measurements."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    cases: int
    valid_outputs: int
    correct_functions: int
    correct_arguments: int
    fully_correct: int
    json_validity: float
    schema_validity: float
    function_accuracy: float
    argument_accuracy: float
    full_accuracy: float
    duration_seconds: float
    peak_memory_mib: float
    meets_subject_targets: bool
    failures: list[str]


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark command-line parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark constrained generation."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--functions_definition", type=Path, default=DEFAULT_DEFINITIONS
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def load_cases(path: Path) -> list[BenchmarkCase]:
    """Load labeled benchmark cases with readable errors."""
    try:
        with path.open(encoding="utf-8") as stream:
            raw: object = json.load(stream)
        cases = TypeAdapter(list[BenchmarkCase]).validate_python(
            raw, strict=True
        )
    except FileNotFoundError as exc:
        raise InputFileError(f"benchmark file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputFileError(
            f"invalid benchmark JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    except ValidationError as exc:
        raise InputFileError(
            f"invalid benchmark cases in {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise InputFileError(
            f"could not read benchmark {path}: {exc}"
        ) from exc
    if not cases:
        raise InputFileError("benchmark must contain at least one case")
    return cases


def build_report(
    model_name: str,
    cases: Sequence[BenchmarkCase],
    results: Sequence[FunctionCallResult | None],
    duration_seconds: float,
    peak_memory_mib: float,
    failures: list[str],
) -> BenchmarkReport:
    """Calculate validity and accuracy rates from independently labeled
    cases."""
    total = len(cases)
    if len(results) != total:
        raise ValueError("benchmark result count must match case count")
    valid_outputs = sum(result is not None for result in results)
    correct_functions = sum(
        result is not None and result.name == case.expected_name
        for case, result in zip(cases, results)
    )
    correct_arguments = sum(
        result is not None and result.parameters == case.expected_parameters
        for case, result in zip(cases, results)
    )
    fully_correct = sum(
        result is not None
        and result.name == case.expected_name
        and result.parameters == case.expected_parameters
        for case, result in zip(cases, results)
    )
    json_validity = valid_outputs / total
    schema_validity = valid_outputs / total
    function_accuracy = correct_functions / total
    argument_accuracy = correct_arguments / total
    full_accuracy = fully_correct / total
    meets_targets = (
        json_validity == 1.0
        and schema_validity == 1.0
        and function_accuracy >= 0.9
        and argument_accuracy >= 0.9
        and duration_seconds < 300.0
    )
    return BenchmarkReport(
        model=model_name,
        cases=total,
        valid_outputs=valid_outputs,
        correct_functions=correct_functions,
        correct_arguments=correct_arguments,
        fully_correct=fully_correct,
        json_validity=json_validity,
        schema_validity=schema_validity,
        function_accuracy=function_accuracy,
        argument_accuracy=argument_accuracy,
        full_accuracy=full_accuracy,
        duration_seconds=round(duration_seconds, 3),
        peak_memory_mib=round(peak_memory_mib, 3),
        meets_subject_targets=meets_targets,
        failures=failures,
    )


def peak_memory_mib() -> float:
    """Return maximum resident memory using platform-specific ru_maxrss
    units."""
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def write_report(path: Path, report: BenchmarkReport) -> None:
    """Write a formatted benchmark report."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(report.model_dump(mode="json"), stream, indent=2)
            stream.write("\n")
    except OSError as exc:
        raise InputFileError(
            f"could not write benchmark report {path}: {exc}"
        ) from exc


def run_benchmark(
    cases: list[BenchmarkCase],
    definitions_path: Path,
) -> BenchmarkReport:
    """Run every labeled case with one model and decoder instance."""
    functions = load_function_definitions(definitions_path)
    client = QwenClient()
    started = time.perf_counter()
    client.load()
    decoder = ConstrainedDecoder.from_client(client)
    results: list[FunctionCallResult | None] = []
    failures: list[str] = []
    for index, case in enumerate(cases, start=1):
        print(f"Benchmark case {index}/{len(cases)}: {case.prompt}")
        try:
            result = decoder.generate(case.prompt, functions)
            results.append(result)
            print(f"  -> {result.name} {result.parameters}")
        except GenerationError as exc:
            results.append(None)
            failures.append(f"case {index}: {exc}")
            print(f"  -> error: {exc}", file=sys.stderr)
    duration = time.perf_counter() - started
    return build_report(
        client.model_name,
        cases,
        results,
        duration,
        peak_memory_mib(),
        failures,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and return failure when subject targets are missed."""
    args = build_parser().parse_args(argv)
    try:
        cases = load_cases(args.cases)
        report = run_benchmark(cases, args.functions_definition)
        write_report(args.report, report)
        print(report.model_dump_json(indent=2))
        return 0 if report.meets_subject_targets else 1
    except (
        InputFileError,
        ModelLoadError,
        VocabularyError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: unexpected benchmark failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
