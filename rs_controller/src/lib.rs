#![allow(clippy::too_many_arguments)]

// Core modules - public for use by binaries and other crates
pub mod action;
pub mod auth;
pub mod core;
pub mod error;
mod informer;
pub mod runtime;

// Python bindings (the flyte SDK's controller extension). Off for pure-Rust
// consumers, which then neither link libpython nor embed an interpreter.
#[cfg(feature = "python")]
mod python;

