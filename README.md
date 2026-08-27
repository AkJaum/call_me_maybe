*This project has been created as part of the 42 curriculum by akjaum.*

# Call Me Maybe

## Description

Call Me Maybe is a schema-constrained function-calling tool for a small language
model. It translates natural-language requests into machine-readable function
calls without executing the functions themselves.

Given this request:

```text
What is the sum of 2 and 3?
```

and a compatible function definition, the program writes:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

The central requirement is reliability. The program does not merely ask the
model to return JSON and hope that it complies. It applies constrained decoding
to every generated token so that invalid JSON, unknown functions, wrong
parameter keys, missing parameters, extra parameters, and schema-invalid scalar
types cannot enter the output.

The required default model is `Qwen/Qwen3-0.6B`. Function selection and argument
extraction come from the model's logits, not from keyword matching, regular
expressions, or hardcoded answers.

## Features

- Schema-constrained token generation.
- Function selection based on Qwen logits.
- Exact parameter-name and scalar-type enforcement.
- Support for `string`, `number`, `integer`, and `boolean` arguments.
- Incremental handling of JSON strings, escapes, Unicode, numbers, and booleans.
- Public `llm_sdk` API usage only.
- Byte-level BPE vocabulary decoding.
- Pydantic validation at input, generation, and output boundaries.
- Atomic output replacement: failed batches do not leave partial JSON.
- Clear errors and non-zero exit codes for invalid inputs or generation failures.
- Reproducible tests and an optional labeled benchmark.
- Optional standalone HTML visualization of every constrained token decision.
- Cached valid-token sets with observable hit/miss diagnostics.

## Project structure

```text
.
├── benchmarks/             # Labeled diagnostic cases and latest measurements
├── data/input/             # Demonstration function definitions and prompts
├── llm_sdk/                # SDK supplied with the subject
├── src/
│   ├── __main__.py         # CLI
│   ├── benchmark.py        # Accuracy/performance benchmark
│   ├── generation.py       # Logit masking and generation loop
│   ├── grammar.py          # Incremental schema-aware JSON grammar
│   ├── io.py               # Validated and atomic file I/O
│   ├── model.py            # Public SDK adapter
│   ├── models.py           # Pydantic contracts
│   ├── pipeline.py         # Multi-prompt application pipeline
│   ├── visualization.py    # Optional safe HTML generation trace
│   └── vocabulary.py       # Byte-level token ID to text mapping and cache
└── tests/                  # Unit and integration tests
```

## Instructions

### Requirements

- Python 3.10 or newer.
- [`uv`](https://docs.astral.sh/uv/).
- Enough disk and memory for `Qwen/Qwen3-0.6B` and its runtime dependencies.
- Network access on the first model run. Later runs can use the local Hugging
  Face cache.

### Installation

Install the locked project dependencies:

```bash
uv sync
```

The local `llm_sdk` package is connected through `pyproject.toml`; it must remain
inside the repository.

### Running

Use the default files:

```bash
uv run python -m src
```

Equivalent Makefile command:

```bash
make run
```

Use custom files:

```bash
uv run python -m src \
  --functions_definition path/to/functions.json \
  --input path/to/prompts.json \
  --output path/to/results.json
```

Validate inputs without loading the model:

```bash
uv run python -m src --validate-only
```

Inspect the public model API, encoded IDs, logit count, and vocabulary path:

```bash
make model-check
```

Generate the mandatory JSON and an optional self-contained HTML token trace:

```bash
uv run python -m src --visualize data/output/generation_trace.html
```

The equivalent convenience command is `make visualize`. The HTML report is a
bonus artifact; it never changes the required JSON path, keys, or contents.

The first full run can download the Qwen weights. An unauthenticated Hugging Face
warning is informational; setting an `HF_TOKEN` is optional and only affects Hub
rate limits.

## Input formats

### Function definitions

The function-definition file is a JSON array:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

Function names must be non-empty and unique. Parameter names must be non-empty.
The mandatory implementation currently accepts scalar parameter types only:
`string`, `number`, `integer`, and `boolean`.

### Prompts

The prompt file is a JSON array:

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet Shrek"}
]
```

Prompt strings must not be empty, and no extra fields are accepted.

## Output format

The default destination is:

```text
data/output/function_calling_results.json
```

It is created during execution and ignored by Git, as required by the subject.
The root is one JSON array. Every entry contains exactly:

- `prompt`: the unchanged input prompt;
- `name`: one declared function name;
- `parameters`: every required argument with the declared type.

There is no prose, comment, trailing comma, or additional key in the file. The
subject version 1.5 names these fields `prompt`, `name`, and `parameters`; this
implementation follows that primary specification.

## Constrained decoding algorithm

### 1. Validate the external contracts

Pydantic validates the complete function catalog and prompt batch before the
model is loaded. Empty names, duplicate functions, unsupported types, malformed
JSON, and unexpected fields fail with readable messages.

### 2. Build the model context

The model receives the natural-language request plus the available function
names, descriptions, parameter names, and parameter types. The prompt is compact
because the decoder, rather than repeated prose, guarantees the output shape.

### 3. Load and decode the public vocabulary

The program obtains `vocab.json` through
`get_path_to_vocab_file()`. Qwen uses byte-level BPE, so strings such as `Ġhello`
are not compared directly against JSON text. The inverse byte-to-Unicode table
converts every vocabulary entry to its real byte sequence and then to UTF-8.

Tokens that contain only an incomplete UTF-8 sequence are excluded. Unicode can
still be emitted through complete UTF-8 tokens or JSON `\uXXXX` escapes.

### 4. Track every valid schema prefix

`FunctionCallGrammar` represents three states: invalid, extendable prefix, and
complete. It keeps every function whose canonical JSON can still match the
generated prefix. A whole token fragment is accepted only when at least one
function schema remains possible.

The grammar enforces:

- canonical object structure;
- a declared function name;
- exact parameter keys and order;
- all required arguments and no extras;
- JSON string and escape rules;
- JSON number and integer syntax;
- lowercase JSON booleans;
- no content after the closing object.

### 5. Mask logits token by token

For every generation step:

1. `get_logits_from_input_ids()` returns the model's next-token logits.
2. Every vocabulary token is tested as a complete fragment against the grammar.
3. Every invalid or non-finite logit is set to negative infinity.
4. The highest remaining logit is selected.
5. Its ID is appended to the model input and its text advances the grammar.

This is deterministic greedy constrained decoding. The model chooses between all
schema-valid alternatives. The decoder only removes impossible alternatives; it
does not choose a function from keywords.

### 6. Stop and validate again

Generation stops only in a complete grammar state. A configurable token limit
prevents unbounded loops, and an empty valid-token set produces a controlled
error. The completed text then passes through `json.loads`, a strict generated
call model, and dynamic validation against the selected function.

The already-known original prompt is attached after generation. The complete
batch is serialized to a temporary file, flushed, synchronized, and atomically
replaces the destination only after every prompt succeeds.

### 7. Optionally visualize the decisions

With `--visualize`, the same generation loop also records the selected token ID,
decoded fragment, selected logit, number of allowed tokens, accumulated valid
prefix, duration, and constrained-token cache counters. These Pydantic-validated
records are rendered as escaped, standalone HTML. Trace collection is disabled
by default, so the mandatory path does not retain per-token diagnostic data.

## Design decisions

### Canonical JSON instead of whitespace-flexible JSON

A single compact representation reduces grammar states and model work while
remaining fully valid JSON. The final file is reformatted for readability.

### Generate only unknown information

The original prompt is not generated by the LLM because the application already
owns it. This avoids copying errors while preserving the exact required output.

### Greedy selection

Argmax after masking is deterministic, reproducible, and sufficient for the
mandatory task. Sampling would add run-to-run variance without improving the
structural guarantee.

### Reject unsupported complex schemas

Nested objects and arrays are a bonus feature in the subject. Accepting their
type names without enforcing their internal schema would create a false safety
guarantee, so they are rejected explicitly.

### Lazy model loading

Input validation and empty batches do not load Qwen. A normal non-empty batch
loads one model, one vocabulary, and one decoder and reuses them for every prompt.

### Atomic writes

Results stay in memory until the batch is complete. A temporary file in the
destination directory prevents an interrupted or failed run from replacing a
previous valid output with partial JSON.

## Error handling

Handled failures include:

- missing or unreadable files;
- malformed JSON with line and column information;
- invalid schemas or prompt records;
- model/SDK loading errors;
- invalid or unreadable vocabularies;
- empty or non-finite logits;
- no valid token continuation;
- generation token-limit exhaustion;
- final schema mismatch;
- output-directory and atomic-write failures.
- optional visualization-write failures.

The CLI prints a concise `error:` message to standard error and returns a non-zero
status. It does not expose an uncontrolled traceback during normal error paths.

## Testing strategy

Run the complete suite:

```bash
make test
```

Run mandatory lint and type checks:

```bash
make lint
```

The current suite contains 58 tests covering:

- valid, malformed, missing, and non-UTF-8 input files;
- strict Pydantic contracts;
- all supported scalar types and numeric edge cases;
- JSON escapes, Unicode, empty strings, large integers, and exponents;
- invalid prefixes, names, keys, order, types, and extra content;
- byte-level BPE conversion and malformed vocabularies;
- real `-inf` masking and higher-scoring invalid tokens;
- empty logits, impossible continuations, and token limits;
- CLI ordering, one-decoder reuse, custom paths, and empty batches;
- atomic output success and failure behavior;
- benchmark loading and score calculations;
- cache counters, trace collection, HTML escaping, and atomic trace writes.

Fast generation tests use a controlled SDK substitute. Separate real-model runs
verify that the same assumptions hold for Qwen's 151,936 logits and public
vocabulary.

## Performance analysis

Measurements were made on the project development machine using CPU inference.
They include model loading and use one model instance per batch.

| Workload | Valid JSON/schema | Function accuracy | Argument accuracy | Time | Peak memory |
|---|---:|---:|---:|---:|---:|
| Default 2-prompt demonstration | 100% | 100% (2/2) | 100% (2/2) | 148.89 s | 5114.86 MiB |
| 4-case diagnostic benchmark | 100% | 100% (4/4) | 100% (4/4) | 348.502 s | 5120.520 MiB |

The default workload satisfies the subject's five-minute target. The harder
four-case diagnostic set exceeds it by 48.502 seconds, so the project does not
claim that this expanded benchmark meets the target. Its machine-readable report
is stored in `benchmarks/latest_results.json`.

Run the same measurement with:

```bash
make benchmark
```

The benchmark returns status 0 only if JSON/schema validity is 100%, function and
argument accuracy are at least 90%, and total duration is below 300 seconds. On
the recorded CPU run it correctly returns status 1 because of time, despite 100%
accuracy and validity.

The main performance constraint is the supplied public SDK interface:
`get_logits_from_input_ids()` recomputes the complete growing sequence for every
new token and does not expose a key/value attention cache. A measured attempt to
prefill grammar-forced characters was removed because repeated full-vocabulary
scans made performance worse. The retained optimization is caching allowed token
IDs by schema, grammar prefix, and logit-vector size.

The optional trace exposes hit/miss counts so this cache behavior can be
demonstrated during peer review instead of being only an implementation claim.

These measurements use only four labeled diagnostic cases and must not be read as
a general accuracy claim. A larger private evaluation is still necessary.

## Challenges faced

### Tokens are not characters

Qwen vocabulary entries may contain spaces, punctuation, several characters, or
byte-level Unicode representations. The decoder therefore validates complete
decoded token fragments rather than assuming one token equals one character.

### JSON numbers have ambiguous prefixes

`-`, `1.`, and `2e+` are valid prefixes but not complete JSON numbers. The
incremental grammar distinguishes extendable prefixes from complete values and
rejects leading zeros, `NaN`, and infinity.

### Python booleans are integers

`bool` subclasses `int` in Python. Explicit exact-type checks prevent `true` from
satisfying an `integer` or `number` parameter.

### Reliability versus model freedom

Over-constraining values would silently choose answers for the model;
under-constraining would violate the schema guarantee. The grammar constrains
structure, declared choices, and types while leaving function branches and value
contents to model logits.

### CPU inference cost

The required token-by-token public SDK API is correct but expensive without an
attention cache. Prompt prose was shortened after measurement while retaining all
function/schema information. A more aggressive forced-prefix experiment was
measured and removed when it regressed runtime.

## Known limitations

- Only scalar arguments are supported; arrays and nested objects are not.
- Independently invalid UTF-8 byte fragments are skipped.
- Greedy decoding has no backtracking if the best valid semantic path is poor.
- Performance scales with generated tokens because the public SDK recomputes the
  full context.
- The labeled benchmark is intentionally small and cannot establish general
  90%+ accuracy.
- The implementation uses the required Qwen model only; multi-model support is
  not claimed.

## Bonus features

The bonus work is deliberately isolated from the mandatory path:

- generation-process visualization through `--visualize` / `make visualize`;
- constrained-token caching with visible hit/miss statistics;
- a comprehensive automated suite, diagnostic benchmark, and atomic recovery
  behavior for both required JSON and optional HTML writes.

Multiple models, a fully reimplemented tokenizer, batching, and nested arguments
are not claimed because they were not implemented and demonstrated end to end.

## Resources

- [Call Me Maybe subject](en.subject.pdf) — primary project specification.
- [RFC 8259: The JSON Data Interchange Format](https://www.rfc-editor.org/info/rfc8259/)
  — normative JSON grammar, including strings and numbers.
- [Python `json` documentation](https://docs.python.org/3/library/json.html) —
  parsing and serialization behavior used at file boundaries.
- [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/) — validated
  data contracts and strict model configuration.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — required model
  and upstream usage information.
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — model-family design
  and capabilities.
- [Hugging Face tokenization algorithms](https://huggingface.co/docs/transformers/main/tokenizer_summary)
  — subword and byte-level BPE background.
- [uv project syncing](https://docs.astral.sh/uv/concepts/projects/sync/) — locked
  dependency and environment workflow.

## Use of AI

AI assistance was used to:

- compare the existing repository with the subject and evaluation scale;
- discuss constrained-decoding architecture and failure modes;
- identify repetitive validation and edge-case test scenarios;
- review type hints, error boundaries, and documentation structure;
- draft and refine this development documentation.

All generated suggestions were inspected against the subject, implemented in
small stages, checked with `flake8` and `mypy`, exercised by automated tests, and
validated with real Qwen runs. AI output was not treated as evidence: performance
and accuracy statements in this README come from recorded local executions.

For a chronological explanation of what changed, why, and how, see
[`Log_de_desenvolvimento.md`](Log_de_desenvolvimento.md).
