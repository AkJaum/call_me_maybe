"""Command-line entry point for Call Me Maybe."""

import argparse
import sys
from pathlib import Path

from src.generation import ConstrainedDecoder, GenerationError
from src.io import (
    InputFileError,
    load_function_definitions,
    load_prompts,
    write_results,
)
from src.model import ModelLoadError, QwenClient
from src.pipeline import generate_results, generate_results_with_traces
from src.visualization import VisualizationError, write_generation_report
from src.vocabulary import VocabularyError

DEFAULT_DEFINITIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate schema-constrained function calls with Qwen3-0.6B."
        )
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_DEFINITIONS,
        help="JSON file containing available function definitions",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="JSON prompt input file",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="result JSON file"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate inputs without loading the language model",
    )
    parser.add_argument(
        "--inspect-model",
        action="store_true",
        help="load Qwen and demonstrate the SDK encode/logits/vocabulary API",
    )
    parser.add_argument(
        "--visualize",
        type=Path,
        help="write an optional standalone HTML trace of token decisions",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate inputs and dispatch the requested project operation."""
    args = build_parser().parse_args(argv)
    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompts(args.input)
        print(
            f"Validated {len(functions)} functions and {len(prompts)} prompts."
        )

        if args.validate_only:
            return 0

        if not prompts:
            if args.inspect_model:
                raise InputFileError(
                    "cannot inspect the model without a prompt"
                )
            write_results(args.output, [])
            if args.visualize is not None:
                write_generation_report(args.visualize, [])
            print(f"Wrote 0 function calls to {args.output}.")
            return 0

        client = QwenClient()
        client.load()
        if args.inspect_model:
            token_ids = client.encode(prompts[0].prompt)
            logits = client.next_token_logits(token_ids)
            print(f"Model: {client.model_name}")
            print(f"Input token IDs: {token_ids}")
            print(f"Logit count: {len(logits)}")
            print(f"Vocabulary: {client.vocab_path()}")
            return 0

        decoder = ConstrainedDecoder.from_client(client)
        if args.visualize is None:
            results = generate_results(decoder, prompts, functions)
        else:
            results, traces = generate_results_with_traces(
                decoder, prompts, functions
            )
            write_generation_report(args.visualize, traces)
            print(f"Wrote generation visualization to {args.visualize}.")
        write_results(args.output, results)
        print(f"Wrote {len(results)} function calls to {args.output}.")
        return 0
    except (
        GenerationError,
        InputFileError,
        ModelLoadError,
        VisualizationError,
        VocabularyError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: unexpected failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
