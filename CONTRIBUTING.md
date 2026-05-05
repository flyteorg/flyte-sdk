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
- Include code and example snippets in function/class docstrings

## Resources

- **[Slack](https://slack.flyte.org/)** — Chat with the community
- **[GitHub Discussions](https://github.com/flyteorg/flyte/discussions)** — Ask questions
- **[Issues](https://github.com/flyteorg/flyte/issues)** — Report bugs
