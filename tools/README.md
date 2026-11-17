# Optional evaluator

The application runs with `uv sync` followed by `uv run python -m src`.
It does not need the Moulinette evaluator to be extracted.

`moulinette.zip` preserves the supplied evaluator source, its own dependency
manifest and lockfile, and the existing `successfully/input/` and
`successfully/correction/` fixtures. The ZIP contains a top-level `moulinette/`
directory. Generated student answers are excluded.

From the repository root:

```bash
make moulinette-prepare
make moulinette-test
```

Preparation extracts the archive only when `moulinette/` is absent and checks
that the two fixture input files exist. It leaves an existing extraction intact.
The test target also prepares the evaluator automatically, generates the private
answers, and runs `grade_student_answers --set private` in the evaluator's own
environment. First use may download evaluator dependencies and model weights.

The complete extracted directory is ignored by Git, including its environment
and `successfully/output/`. To restore a fresh copy, move any modified extraction
elsewhere, then rerun `make moulinette-prepare`.

The SDK stays extracted at `llm_sdk/`: the subject requires that directory and
installation with `uv sync` alone. Its runtime source is unchanged; the root
`uv.lock` locks the application dependencies. Duplicate SDK and example-data
archives are unnecessary because those files are already available directly.
