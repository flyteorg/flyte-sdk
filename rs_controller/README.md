# flyte_core

The Flyte controller core: submit actions to the Flyte ActionsService, track them
through a watch-backed informer cache, and authenticate against a Flyte
deployment.

This crate backs two consumers:

- the **Python** `flyte_controller_base` extension module, built from this same
  source with maturin and used by `flyte` when `_F_USE_RUST_CONTROLLER=1`;
- the **Rust SDK** ([`flyte`](https://github.com/flyteorg/flyte-sdk-rs)), which
  drives it directly.

## Two things that will surprise you

**The library is named `flyte_controller_base`, not `flyte_core`.** The crates.io
package name and the library name differ, so imports read:

```rust
use flyte_controller_base::core::CoreBaseController;
use flyte_controller_base::action::{Action, ActionType};
```

**Default features embed a Python interpreter.** `default = ["pyo3/auto-initialize"]`
means a plain `cargo build` links `libpython`, and you need `python3-dev` (or
equivalent) installed to build. That default exists because the crate's own
binaries need it.

`pyo3/auto-initialize` is mutually exclusive with `pyo3/extension-module`, so a
downstream crate that also builds a Python extension module must not take this
one with default features. Building the Python wheel does exactly that — see
`pyproject.toml`, which sets `no-default-features = true` and
`features = ["extension-module"]`.

Note that turning pyo3 off entirely is not currently possible: `flyteidl2`, which
supplies every protobuf type crossing this API, depends on pyo3 unconditionally
and annotates its generated messages as `#[pyclass]`. Separating a pure-Rust core
from the pyo3 bindings is planned, and it will be a breaking release.

## Usage

```toml
[dependencies]
flyte_core = "0.1"
```

```rust
use flyte_controller_base::core::CoreBaseController;

// Auth from _UNION_EAGER_API_KEY. Must be called off an async thread: the
// constructor blocks on the shared runtime.
let controller = CoreBaseController::new_with_auth(20)?;
```

For an unauthenticated endpoint (devbox or local backend), use
`CoreBaseController::new_without_auth(endpoint, workers)`.

## Releasing

Tag `rs-v<version>` to publish to crates.io; see
`.github/workflows/publish-rust-crate.yml`. The crate version lives in
`Cargo.toml` and is independent of the Python SDK's version — the workflow checks
the tag against the manifest rather than rewriting it.

## License

Apache-2.0
