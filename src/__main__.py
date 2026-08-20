"""Command-line entry point for Call Me Maybe."""

import argparse
import sys
from pathlib import Path

from src.io import InputFileError, load_function_definitions, load_prompts
from src.model import ModelLoadError, QwenClient


DEFAULT_DEFINITIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate schema-constrained function calls with Qwen3-0.6B."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_DEFINITIONS,
        help="JSON file containing available function definitions",
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="JSON prompt input file"
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
    return parser


def main() -> int:
    """Validate inputs and dispatch the requested project operation."""
    args = build_parser().parse_args()
    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompts(args.input)
        print(f"Validated {len(functions)} functions and {len(prompts)} prompts.")

        if args.validate_only:
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

        print(
            "error: constrained generation is the next implementation milestone; "
            "use --validate-only to verify the current scaffold.",
            file=sys.stderr,
        )
        return 2
    except (InputFileError, ModelLoadError, IndexError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
