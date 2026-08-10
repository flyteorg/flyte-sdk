# Contributing to Flyte 2

We welcome contributions! Whether it's bug fixes, new features, documentation improvements, or testing enhancements.

## Setup

```bash
uv sync
make dist
```

This installs the package in editable mode and builds a wheel so the default `Image()` uses your local changes. Requires a Docker daemon.

## Guidelines

### Module structure

- **`flyte.*`** — Task authoring experience only
- **`flyte.apps.*`** — App authoring experience
- **`flyte.io.*`** — Flyte special types that perform large I/O
- **`_internal`** — Internal use only

### Keep the core small

- Extensions and extra functionality go in **plugins**, not core
- Maintain clear module separation so that module loading is fast and efficient

### Public API surface

- Users should never need to import `_module` (underscore-prefixed) modules
- Use `__all__` and `__init__.py` to export the public API
- Never expose protobuf to users
- Plugins should also avoid depending on `_modules` — they may change without notice

### Code quality

- `make fmt` — format code
- `make mypy` — type check
- `make check-docstrings` — check docstring style (see below)
- Include code and example snippets in function/class docstrings

### Docstring style

Docstrings are read in three places: the published API reference, an IDE
tooltip, and `help()`. Only the first renders anything but plain text, so
markup aimed at a generator we do not run is visible verbatim to everyone.
This repo has no Sphinx, so reStructuredText in a docstring is inert.

Write **Markdown prose with Google-style sections**. `make check-docstrings`
enforces this and runs in CI.

- **Sections** — `Args:`, `Returns:`, `Raises:`, `Note:`, `Example:`. Entries
  are indented beneath the header as `name: description`, continuation lines
  indented one level further. NumPy sections (a header over a dashed rule) are
  rejected: the API-reference generator does not parse them, so every parameter
  description silently disappears from the rendered table.

  ```python
  Args:
      name: Stable agent identifier.
      instructions: Base system prompt. Skills and a tool catalog summary
          are appended automatically.
  ```

- **References to other symbols** — a plain code span, qualified when the
  symbol is public: `` `flyte.io.Dir` ``, `` `Dir.write_text` ``. The docs site
  turns those into links automatically. Do not use `:class:` / `:meth:` /
  `:func:` roles; they have no effect and render literally.

- **Code blocks** — a fenced block with an explicit language. Do not use the
  RST `::` literal-block marker: it renders as a stray double colon and leaves
  the block with no language, so nothing highlights.

  ````python
  ```python
  d = Dir.new_remote("output")
  ```
  ````

- **Directives** — no `.. warning::`, `.. code-block::`, `.. autosummary::`.
  Use plain prose or a fenced block.

- **Parameter documentation** — use `Args:`, not Sphinx field lists
  (`:param x:`, `:returns:`, `:rtype:`, `:raises X:`). The generator does parse
  field lists, so nothing renders wrong, but they are still reStructuredText and
  read as markup everywhere else a docstring is shown.

- **Inline literals** — a single-backtick code span, `` `value` ``, not the RST
  double-backtick form ` ``value`` `. The double form renders correctly only by
  accident, because a double backtick also happens to be a valid Markdown code
  span.

- **Links** — a Markdown link, `[text](url)`, not an RST hyperlink target
  (`` `text <url>`_ ``).

- **Tables** — a Markdown table. RST grid tables (`+----+----+`) render as
  garbage.

Anything inside a fenced code block is treated as code and is never checked, so
an example that deliberately shows RST is fine.

## Resources

- **[Slack](https://slack.flyte.org/)** — Chat with the community
- **[GitHub Discussions](https://github.com/flyteorg/flyte/discussions)** — Ask questions
- **[Issues](https://github.com/flyteorg/flyte/issues)** — Report bugs
