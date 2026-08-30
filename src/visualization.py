"""Self-contained HTML visualization for constrained generation traces."""

from html import escape
import os
from pathlib import Path
import tempfile

from src.generation import GenerationTrace


class VisualizationError(RuntimeError):
    """Report a controlled failure while writing a trace visualization."""


def render_generation_report(traces: list[GenerationTrace]) -> str:
    """Render safe standalone HTML for a collection of generation traces."""
    sections = "".join(_render_trace(trace) for trace in traces)
    if not sections:
        sections = '<p class="empty">No prompts were provided.</p>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Call Me Maybe — constrained decoding trace</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0 auto; max-width: 1100px; padding: 2rem; }}
    article {{ border: 1px solid #8886; border-radius: .75rem;
      margin: 1.5rem 0; padding: 1rem; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: .75rem 1.5rem; }}
    table {{ border-collapse: collapse; display: block; overflow-x: auto;
      width: 100%; }}
    th, td {{ border-bottom: 1px solid #8885; padding: .5rem; text-align: left;
      vertical-align: top; }}
    code, pre {{ font-family: ui-monospace, monospace; }}
    pre {{ margin: 0; max-width: 44rem; overflow-wrap: anywhere;
      white-space: pre-wrap; }}
    .empty {{ border: 1px dashed #888; padding: 1rem; }}
  </style>
</head>
<body>
  <h1>Constrained decoding trace</h1>
  <p>Every row is a token selected after invalid logits were masked.</p>
  {sections}
</body>
</html>
"""


def write_generation_report(path: Path, traces: list[GenerationTrace]) -> None:
    """Atomically write a trace report, preserving an older file on failure."""
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(render_generation_report(traces))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        message = f"could not write visualization {path}: {exc}"
        raise VisualizationError(message) from exc


def _render_trace(trace: GenerationTrace) -> str:
    """Render one trace using escaped input and generated text."""
    rows = "".join(
        "<tr>"
        f"<td>{step.index}</td>"
        f"<td>{step.token_id}</td>"
        f"<td><code>{escape(repr(step.token_fragment))}</code></td>"
        f"<td>{step.allowed_token_count}</td>"
        f"<td>{step.selected_logit:.6g}</td>"
        f"<td><pre>{escape(step.prefix)}</pre></td>"
        "</tr>"
        for step in trace.steps
    )
    result = escape(trace.result.model_dump_json(indent=2))
    cache_summary = f"{trace.cache_hits} hits / {trace.cache_misses} misses"
    return f"""<article>
  <h2>{escape(trace.prompt)}</h2>
  <div class="summary">
    <span><strong>Model:</strong> {escape(trace.model_name)}</span>
    <span><strong>Duration:</strong> {trace.duration_seconds:.3f}s</span>
    <span><strong>Tokens:</strong> {len(trace.steps)}</span>
    <span><strong>Cache:</strong> {cache_summary}</span>
  </div>
  <h3>Token decisions</h3>
  <table>
    <thead><tr><th>Step</th><th>ID</th><th>Fragment</th><th>Allowed</th>
      <th>Logit</th><th>Valid prefix after selection</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h3>Validated result</h3>
  <pre>{result}</pre>
</article>"""
