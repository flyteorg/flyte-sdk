#![allow(clippy::too_many_arguments)]

// Core modules - public for use by binaries and other crates
pub mod action;
pub mod auth;
pub mod core;
pub mod error;
mod informer;
pub mod proto;

// Python bindings - thin wrappers around core types
use std::sync::Arc;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_async_runtimes::tokio::future_into_py;
use tracing::{error, info, warn};
use tracing_subscriber::FmtSubscriber;

use crate::action::{Action, ActionType};
use crate::core::CoreBaseController;
use crate::error::ControllerError;
use flyteidl2::flyteidl::common::{ActionIdentifier, RunIdentifier};
use prost::Message;

// Python error conversions
impl From<ControllerError> for PyErr {
    fn from(err: ControllerError) -> Self {
        exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

impl From<crate::auth::AuthConfigError> for PyErr {
    fn from(err: crate::auth::AuthConfigError) -> Self {
        exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

/// Base class for RemoteController to eventually inherit from
#[pyclass(subclass)]
struct BaseController(Arc<CoreBaseController>);

#[pymethods]
impl BaseController {
    #[new]
    #[pyo3(signature = (*, endpoint=None))]
    fn new(endpoint: Option<String>) -> PyResult<Self> {
        let core_base = if let Some(ep) = endpoint {
            info!("Creating controller wrapper with endpoint {:?}", ep);
            CoreBaseController::new_without_auth(ep)?
        } else {
            info!("Creating controller wrapper from _UNION_EAGER_API_KEY env var");
            CoreBaseController::new_with_auth()?
        };
        Ok(BaseController(core_base))
    }

    /// `async def submit(self, action: Action) -> Action`
    ///
    /// Enqueue `action`.
    fn submit_action<'py>(&self, py: Python<'py>, action: Action) -> PyResult<Bound<'py, PyAny>> {
        let real_base = self.0.clone();
        let py_fut = future_into_py(py, async move {
            let action_id = action.action_id.clone();
            real_base.submit_action(action).await.map_err(|e| {
                error!("Error submitting action {:?}: {:?}", action_id, e);
                exceptions::PyRuntimeError::new_err(format!("Failed to submit action: {}", e))
            })
        });
        py_fut
    }

    fn cancel_action<'py>(&self, py: Python<'py>, action: Action) -> PyResult<Bound<'py, PyAny>> {
        let real_base = self.0.clone();
        let mut a = action.clone();
        let py_fut = future_into_py(py, async move {
            real_base.cancel_action(&mut a).await.map_err(|e| {
                error!("Error cancelling action {:?}: {:?}", action.action_id, e);
                exceptions::PyRuntimeError::new_err(format!("Failed to cancel action: {}", e))
            })
        });
        py_fut
    }

    fn get_action<'py>(
        &self,
        py: Python<'py>,
        // action_id: ActionIdentifier,
        action_id_bytes: &[u8],
        parent_action_name: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let real_base = self.0.clone();
        let action_id = ActionIdentifier::decode(action_id_bytes).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to decode ActionIdentifier: {}",
                e
            ))
        })?;
        let py_fut = future_into_py(py, async move {
            real_base
                .get_action(action_id.clone(), parent_action_name.as_str())
                .await
                .map_err(|e| {
                    error!("Error getting action {:?}: {:?}", action_id, e);
                    exceptions::PyRuntimeError::new_err(format!("Failed to cancel action: {}", e))
                })
        });
        py_fut
    }

    fn finalize_parent_action<'py>(
        &self,
        py: Python<'py>,
        // run_id: RunIdentifier,
        run_id_bytes: &[u8],
        parent_action_name: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let base = self.0.clone();
        let parent_action_string = parent_action_name.to_string();
        let run_id = RunIdentifier::decode(run_id_bytes).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to decode RunIdentifier: {}",
                e
            ))
        })?;
        let py_fut = future_into_py(py, async move {
            base.finalize_parent_action(&run_id, &parent_action_string)
                .await;
            warn!("Parent action finalize: {}", parent_action_string);
            Ok(())
        });
        py_fut
    }

    fn watch_for_errors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let base = self.0.clone();
        let py_fut = future_into_py(py, async move {
            base.watch_for_errors().await.map_err(|e| {
                error!("Controller watch_for_errors detected failure: {:?}", e);
                exceptions::PyRuntimeError::new_err(format!(
                    "Controller watch ended with failure: {}",
                    e
                ))
            })
        });
        py_fut
    }
}

#[pymodule]
fn flyte_controller_base(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // Check if running remotely by checking if FLYTE_INTERNAL_EXECUTION_PROJECT is set
        let is_remote = std::env::var("FLYTE_INTERNAL_EXECUTION_PROJECT").is_ok();
        let is_rich_logging_disabled = std::env::var("DISABLE_RICH_LOGGING").is_ok();
        let disable_ansi = is_remote || is_rich_logging_disabled;

        let subscriber = FmtSubscriber::builder()
            .with_max_level(tracing::Level::DEBUG)
            .with_ansi(!disable_ansi)
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("Failed to set global tracing subscriber");
    });

    m.add_class::<BaseController>()?;
    m.add_class::<Action>()?;
    m.add_class::<ActionType>()?;
    Ok(())
}
