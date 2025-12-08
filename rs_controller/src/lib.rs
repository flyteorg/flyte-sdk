#![allow(clippy::too_many_arguments)]

// Core modules - public for use by binaries and other crates
pub mod action;
pub mod auth;
pub mod core;
mod informer;
pub mod proto;

// Python bindings - thin wrappers around core types
use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_async_runtimes::tokio::future_into_py;
use tower::ServiceBuilder;
use tracing::{error, info, warn};
use tracing_subscriber::FmtSubscriber;

use crate::action::{Action, ActionType};
use crate::auth::{AuthConfig, AuthLayer, ClientCredentialsAuthenticator};
use crate::core::{ControllerError, CoreBaseController};
use flyteidl2::flyteidl::common::{ActionIdentifier, ProjectIdentifier};
use flyteidl2::flyteidl::task::task_service_client::TaskServiceClient;
use flyteidl2::flyteidl::task::{list_tasks_request, ListTasksRequest};
use flyteidl2::flyteidl::workflow::state_service_client::StateServiceClient;
use tonic::transport::Endpoint;

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

    #[staticmethod]
    fn try_list_tasks(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        future_into_py(py, async move {
            use flyteidl2::flyteidl::common::ListRequest;
            use tonic::Code;

            let api_key = std::env::var("_UNION_EAGER_API_KEY").unwrap_or_else(|_| {
                warn!("_UNION_EAGER_API_KEY env var not set, using empty string");
                String::new()
            });

            let auth_config = AuthConfig::new_from_api_key(api_key.as_str())?;
            let endpoint = auth_config.endpoint.clone();
            let static_endpoint = endpoint.clone().leak();
            // Strip "https://" (8 chars) to get just the hostname for TLS config
            let domain = endpoint.strip_prefix("https://").ok_or_else(|| {
                ControllerError::SystemError("Endpoint must start with https://".to_string())
            })?;
            let endpoint = crate::core::create_tls_endpoint(static_endpoint, domain).await?;
            let channel = endpoint.connect().await.map_err(ControllerError::from)?;

            let authenticator = Arc::new(ClientCredentialsAuthenticator::new(auth_config));

            let auth_handling_channel = ServiceBuilder::new()
                .layer(AuthLayer::new(authenticator, channel.clone()))
                .service(channel);

            let mut task_client = TaskServiceClient::new(auth_handling_channel);

            let list_request_base = ListRequest {
                limit: 100,
                ..Default::default()
            };
            let req = ListTasksRequest {
                request: Some(list_request_base),
                known_filters: vec![],
                scope_by: Some(list_tasks_request::ScopeBy::ProjectId(ProjectIdentifier {
                    organization: "demo".to_string(),
                    domain: "development".to_string(),
                    name: "flytesnacks".to_string(),
                })),
            };

            let mut attempts = 0;
            let final_result = loop {
                let result = task_client.list_tasks(req.clone()).await;
                match result {
                    Ok(response) => {
                        println!("Success: {:?}", response.into_inner());
                        break Ok(true);
                    }
                    Err(status) if status.code() == Code::Unauthenticated && attempts < 1 => {
                        attempts += 1;
                        continue;
                    }
                    Err(status) => {
                        eprintln!("Error calling gRPC: {}", status);
                        break Err(exceptions::PyRuntimeError::new_err(format!(
                            "gRPC error: {}",
                            status
                        )));
                    }
                }
            };
            warn!("Finished try_list_tasks with result {:?}", final_result);
            final_result
        })
    }

    #[staticmethod]
    fn try_watch(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        future_into_py(py, async move {
            use flyteidl2::flyteidl::common::ActionIdentifier;
            use flyteidl2::flyteidl::common::RunIdentifier;
            use flyteidl2::flyteidl::workflow::watch_request::Filter;
            use flyteidl2::flyteidl::workflow::WatchRequest;
            use tokio::time::sleep;

            info!("Starting watch example with authentication and retry...");

            // Read in the api key which gives us the endpoint to connect to as well as the credentials
            let api_key = std::env::var("_UNION_EAGER_API_KEY").unwrap_or_else(|_| {
                warn!("_UNION_EAGER_API_KEY env var not set, using empty string");
                String::new()
            });

            let auth_config = AuthConfig::new_from_api_key(api_key.as_str())?;
            let endpoint = auth_config.endpoint.clone();
            let static_endpoint = endpoint.clone().leak();
            // Strip "https://" (8 chars) to get just the hostname for TLS config
            let domain = endpoint.strip_prefix("https://").ok_or_else(|| {
                ControllerError::SystemError("Endpoint must start with https://".to_string())
            })?;
            let endpoint = crate::core::create_tls_endpoint(static_endpoint, domain).await?;
            let channel = endpoint.connect().await.map_err(ControllerError::from)?;

            let authenticator = Arc::new(ClientCredentialsAuthenticator::new(auth_config));

            // Wrap channel with auth layer - ALL calls now automatically authenticated!
            let auth_channel = ServiceBuilder::new()
                .layer(AuthLayer::new(authenticator, channel.clone()))
                .service(channel);

            let mut client = StateServiceClient::new(auth_channel);

            // Watch configuration (matching Python example)
            let run_id = RunIdentifier {
                org: "demo".to_string(),
                project: "flytesnacks".to_string(),
                domain: "development".to_string(),
                name: "r57jklb4mw4k6bkb2p88".to_string(),
            };
            let parent_action_name = "a0".to_string();

            // Retry parameters (matching Python defaults)
            let min_watch_backoff = Duration::from_secs(1);
            let max_watch_backoff = Duration::from_secs(30);
            let max_watch_retries = 10;

            // Watch loop with retry logic (following Python _informer.py pattern)
            let mut retries = 0;
            let mut message_count = 0;

            while retries < max_watch_retries {
                if retries >= 1 {
                    warn!("Watch retrying, attempt {}/{}", retries, max_watch_retries);
                }

                // Create watch request
                let request = WatchRequest {
                    filter: Some(Filter::ParentActionId(ActionIdentifier {
                        name: parent_action_name.clone(),
                        run: Some(run_id.clone()),
                    })),
                };

                // Establish the watch stream
                // The outer retry loop handles failures, middleware handles auth refresh
                let stream_result = client.watch(request.clone()).await;

                match stream_result {
                    Ok(response) => {
                        info!("Successfully established watch stream");
                        let mut stream = response.into_inner();

                        // Process messages from the stream
                        loop {
                            match stream.message().await {
                                Ok(Some(watch_response)) => {
                                    // Successfully received a message - reset retry counter
                                    retries = 0;
                                    message_count += 1;

                                    // Process the message (enum with ActionUpdate or ControlMessage)
                                    use flyteidl2::flyteidl::workflow::watch_response::Message;
                                    match &watch_response.message {
                                        Some(Message::ControlMessage(control_msg)) => {
                                            if control_msg.sentinel {
                                                info!(
                                                    "Received Sentinel for parent action: {}",
                                                    parent_action_name
                                                );
                                            }
                                        }
                                        Some(Message::ActionUpdate(action_update)) => {
                                            info!(
                                                "Received action update for: {} (phase: {:?})",
                                                action_update
                                                    .action_id
                                                    .as_ref()
                                                    .map(|id| id.name.as_str())
                                                    .unwrap_or("unknown"),
                                                action_update.phase
                                            );

                                            if !action_update.output_uri.is_empty() {
                                                info!("Output URI: {}", action_update.output_uri);
                                            }

                                            if action_update.phase == 4 {
                                                // PHASE_FAILED
                                                if action_update.error.is_some() {
                                                    error!(
                                                        "Action failed with error: {:?}",
                                                        action_update.error
                                                    );
                                                }
                                            }
                                        }
                                        None => {
                                            warn!("Received empty watch response");
                                        }
                                    }

                                    // For demo purposes, exit after receiving a few messages
                                    if message_count >= 50 {
                                        info!("Received {} messages, exiting demo", message_count);
                                        return Ok(true);
                                    }
                                }
                                Ok(None) => {
                                    warn!("Watch stream ended gracefully");
                                    break; // Stream ended, retry
                                }
                                Err(status) => {
                                    error!("Error receiving message from watch stream: {}", status);

                                    // Check if it's an auth error
                                    if status.code() == tonic::Code::Unauthenticated {
                                        warn!("Unauthenticated error - credentials will be refreshed on retry");
                                    }

                                    break; // Break inner loop to retry
                                }
                            }
                        }
                    }
                    Err(status) => {
                        error!("Failed to establish watch stream: {}", status);

                        if status.code() == tonic::Code::Unauthenticated {
                            warn!("Unauthenticated error - credentials will be refreshed on retry");
                        }
                    }
                }

                // Increment retry counter and apply exponential backoff
                retries += 1;
                if retries < max_watch_retries {
                    let backoff = min_watch_backoff
                        .saturating_mul(2_u32.pow(retries as u32))
                        .min(max_watch_backoff);
                    warn!("Watch failed, retrying in {:?}...", backoff);
                    sleep(backoff).await;
                }
            }

            // Exceeded max retries
            error!(
                "Watch failure retries crossed threshold {}/{}, exiting!",
                retries, max_watch_retries
            );
            Err(exceptions::PyRuntimeError::new_err(format!(
                "Max watch retries ({}) exceeded",
                max_watch_retries
            )))
        })
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
        action_id: ActionIdentifier,
    ) -> PyResult<Bound<'py, PyAny>> {
        let real_base = self.0.clone();
        let py_fut = future_into_py(py, async move {
            real_base.get_action(action_id.clone()).await.map_err(|e| {
                error!("Error getting action {:?}: {:?}", action_id, e);
                exceptions::PyRuntimeError::new_err(format!("Failed to cancel action: {}", e))
            })
        });
        py_fut
    }
}

#[pymodule]
fn flyte_controller_base(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let subscriber = FmtSubscriber::builder()
            .with_max_level(tracing::Level::DEBUG)
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("Failed to set global tracing subscriber");
    });

    m.add_class::<BaseController>()?;
    m.add_class::<Action>()?;
    m.add_class::<ActionType>()?;
    Ok(())
}
