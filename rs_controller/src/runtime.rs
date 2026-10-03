//! The shared tokio runtime the controller's background workers run on.
//!
//! With the `python` feature it is pyo3-async-runtimes' runtime, so futures
//! handed to Python and the controller's own tasks share one runtime. Without
//! it, a plain process-wide multi-threaded runtime.

#[cfg(feature = "python")]
pub fn get_runtime() -> &'static tokio::runtime::Runtime {
    pyo3_async_runtimes::tokio::get_runtime()
}

#[cfg(not(feature = "python"))]
pub fn get_runtime() -> &'static tokio::runtime::Runtime {
    static RT: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build the flyte_core tokio runtime")
    })
}
