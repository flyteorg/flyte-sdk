#![allow(clippy::too_many_arguments)]

mod action;
pub mod auth; // Public for use in other crates
mod informer;
pub mod proto; // Public for use in other crates

use std::default;
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::time::Duration;

use futures::TryFutureExt;
use pyo3::prelude::*;
use tokio::sync::mpsc;
use tower::ServiceExt;
use tracing::{debug, error, info, warn};

use thiserror::Error;

use crate::action::{Action, ActionType};
use crate::informer::Informer;

use crate::auth::{AuthConfig, AuthConfigError, AuthLayer, ClientCredentialsAuthenticator};
use flyteidl2::flyteidl::common::{ActionIdentifier, ProjectIdentifier};
use flyteidl2::flyteidl::task::task_service_client::TaskServiceClient;
use flyteidl2::flyteidl::task::TaskIdentifier;
use flyteidl2::flyteidl::task::{list_tasks_request, ListTasksRequest};
use flyteidl2::flyteidl::workflow::enqueue_action_request;
use flyteidl2::flyteidl::workflow::queue_service_client::QueueServiceClient;
use flyteidl2::flyteidl::workflow::state_service_client::StateServiceClient;
use flyteidl2::flyteidl::workflow::{EnqueueActionRequest, EnqueueActionResponse, TaskAction, WatchRequest, WatchResponse};
use flyteidl2::google;
use google::protobuf::StringValue;
use pyo3::exceptions;
use pyo3::types::PyAny;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_async_runtimes::tokio::get_runtime;
use tokio::sync::oneshot;
use tokio::sync::OnceCell;
use std::sync::OnceLock;
use tokio::time::sleep;
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint};
use tonic::Status;
use tracing_subscriber::FmtSubscriber;

// Fetches Amazon root CA certificate from Amazon Trust Services
async fn fetch_amazon_root_ca() -> Result<Certificate, ControllerError> {
    // Amazon Root CA 1 - the main root used by AWS services
    let url = "https://www.amazontrust.com/repository/AmazonRootCA1.pem";

    let response = reqwest::get(url)
        .await
        .map_err(|e| ControllerError::SystemError(format!("Failed to fetch certificate: {}", e)))?;

    let cert_pem = response
        .text()
        .await
        .map_err(|e| ControllerError::SystemError(format!("Failed to read certificate: {}", e)))?;

    Ok(Certificate::from_pem(cert_pem))
}

// Helper to create TLS-configured endpoint with Amazon CA certificate
// todo: when we resolve the pem issue, also remove the need to have both inputs which are basically the same
async fn create_tls_endpoint(url: &'static str, domain: &str) -> Result<Endpoint, ControllerError> {
    // Fetch Amazon root CA dynamically
    let cert = fetch_amazon_root_ca().await?;

    let tls_config = ClientTlsConfig::new()
        .domain_name(domain)
        .ca_certificate(cert);

    let endpoint = Endpoint::from_static(url)
        .tls_config(tls_config)
        .map_err(|e| ControllerError::SystemError(format!("TLS config error: {}", e)))?;

    Ok(endpoint)
}

#[derive(Error, Debug)]
pub enum ControllerError {
    #[error("Bad context: {0}")]
    BadContext(String),
    #[error("Runtime error: {0}")]
    RuntimeError(String),
    #[error("System error: {0}")]
    SystemError(String),
    #[error("gRPC error: {0}")]
    GrpcError(#[from] tonic::Status),
    #[error("Task error: {0}")]
    TaskError(String),
}

impl From<tonic::transport::Error> for ControllerError {
    fn from(err: tonic::transport::Error) -> Self {
        ControllerError::SystemError(format!("Transport error: {:?}", err))
    }
}

impl From<ControllerError> for PyErr {
    // can better map errors in the future
    fn from(err: ControllerError) -> Self {
        exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

impl From<AuthConfigError> for PyErr {
    fn from(err: AuthConfigError) -> Self {
        exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

enum ChannelType {
    Plain(tonic::transport::Channel),
    Authenticated(crate::auth::AuthService<tonic::transport::Channel>),
}

#[derive(Clone, Debug)]
pub enum StateClient {
    Plain(StateServiceClient<tonic::transport::Channel>),
    Authenticated(StateServiceClient<crate::auth::AuthService<tonic::transport::Channel>>),
}

impl StateClient {
    pub async fn watch(
        &mut self,
        request: impl tonic::IntoRequest<WatchRequest>,
    ) -> Result<tonic::Response<tonic::codec::Streaming<WatchResponse>>, tonic::Status> {
        match self {
            StateClient::Plain(client) => client.watch(request).await,
            StateClient::Authenticated(client) => client.watch(request).await,
        }
    }
}

#[derive(Clone, Debug)]
pub enum QueueClient {
    Plain(QueueServiceClient<tonic::transport::Channel>),
    Authenticated(QueueServiceClient<crate::auth::AuthService<tonic::transport::Channel>>),
}

impl QueueClient {
    pub async fn enqueue_action(
        &mut self,
        request: impl tonic::IntoRequest<EnqueueActionRequest>,
    ) -> Result<tonic::Response<EnqueueActionResponse>, tonic::Status> {
        match self {
            QueueClient::Plain(client) => client.enqueue_action(request).await,
            QueueClient::Authenticated(client) => client.enqueue_action(request).await,
        }
    }
}

struct CoreBaseController {
    channel: ChannelType,
    informer: OnceCell<Arc<Informer>>,
    state_client_cache: OnceLock<StateClient>,
    queue_client_cache: OnceLock<QueueClient>,
    shared_queue: mpsc::Sender<Action>,
    rx_of_shared_queue: Arc<tokio::sync::Mutex<mpsc::Receiver<Action>>>,
}

impl CoreBaseController {
    // Helper methods to get cached clients (constructed once, reused thereafter)
    fn state_client(&self) -> StateClient {
        self.state_client_cache.get_or_init(|| {
            match &self.channel {
                ChannelType::Plain(ch) => StateClient::Plain(StateServiceClient::new(ch.clone())),
                ChannelType::Authenticated(ch) => StateClient::Authenticated(StateServiceClient::new(ch.clone())),
            }
        }).clone()
    }

    fn queue_client(&self) -> QueueClient {
        self.queue_client_cache.get_or_init(|| {
            match &self.channel {
                ChannelType::Plain(ch) => QueueClient::Plain(QueueServiceClient::new(ch.clone())),
                ChannelType::Authenticated(ch) => QueueClient::Authenticated(QueueServiceClient::new(ch.clone())),
            }
        }).clone()
    }

    pub fn new_with_auth() -> Result<Arc<Self>, ControllerError> {
        use crate::auth::{AuthConfig, AuthLayer, ClientCredentialsAuthenticator};
        use tower::ServiceBuilder;

        info!("Creating CoreBaseController from _UNION_EAGER_API_KEY env var (with auth)");
        // Read from env var and use auth
        let api_key = std::env::var("_UNION_EAGER_API_KEY")
            .map_err(|_| ControllerError::SystemError(
                "_UNION_EAGER_API_KEY env var must be provided".to_string()
            ))?;
        let auth_config = AuthConfig::new_from_api_key(&api_key).expect("Bad api key");
        let endpoint_url = auth_config.endpoint.clone();

        let endpoint_static: &'static str = Box::leak(Box::new(endpoint_url.clone().into_boxed_str()));
        // shared queue
        let (shared_tx, rx_of_shared_queue) = mpsc::channel::<Action>(64);

        let rt = get_runtime();
        let channel = rt.block_on(async {
            // todo: escape hatch for localhost
            // Strip "https://" to get just the hostname for TLS config
            let domain = endpoint_url.strip_prefix("https://")
                .ok_or_else(|| ControllerError::SystemError("Endpoint must start with https:// when using auth".to_string()))?;

            // Create TLS-configured endpoint
            let endpoint = create_tls_endpoint(endpoint_static, domain).await?;
            let channel = endpoint.connect().await.map_err(ControllerError::from)?;

            let authenticator = Arc::new(ClientCredentialsAuthenticator::new(auth_config.clone()));
            let auth_channel = ServiceBuilder::new()
                .layer(AuthLayer::new(authenticator, channel.clone()))
                .service(channel);

            Ok::<_, ControllerError>(ChannelType::Authenticated(auth_channel))
        })?;

        let real_base_controller = CoreBaseController {
            channel,
            informer: OnceCell::new(),
            state_client_cache: OnceLock::new(),
            queue_client_cache: OnceLock::new(),
            shared_queue: shared_tx,
            rx_of_shared_queue: Arc::new(tokio::sync::Mutex::new(rx_of_shared_queue)),
        };

        let real_base_controller = Arc::new(real_base_controller);
        // Start the background worker
        let controller_clone = real_base_controller.clone();
        rt.spawn(async move {
            controller_clone.bg_worker().await;
        });
        Ok(real_base_controller)
    }

    pub fn new_without_auth(endpoint: String) -> Result<Arc<Self>, ControllerError> {
        let endpoint_static: &'static str = Box::leak(Box::new(endpoint.clone().into_boxed_str()));
        // shared queue
        let (shared_tx, rx_of_shared_queue) = mpsc::channel::<Action>(64);

        let rt = get_runtime();
        let channel = rt.block_on(async {
            let chan = if endpoint.starts_with("http://") {
                let endpoint = Endpoint::from_static(endpoint_static);
                endpoint.connect().await.map_err(ControllerError::from)?
            } else if endpoint.starts_with("https://") {
                // Strip "https://" to get just the hostname for TLS config
                let domain = endpoint.strip_prefix("https://")
                    .ok_or_else(|| ControllerError::SystemError("Endpoint must start with https://".to_string()))?;

                // Create TLS-configured endpoint
                let endpoint = create_tls_endpoint(endpoint_static, domain).await?;
                endpoint.connect().await.map_err(ControllerError::from)?
            }
            else {
                return Err(ControllerError::SystemError(format!("Malformed endpoint {}", endpoint)));
            };
            Ok::<_, ControllerError>(ChannelType::Plain(chan))
        })?;

        let real_base_controller = CoreBaseController {
            channel,
            informer: OnceCell::new(),
            state_client_cache: OnceLock::new(),
            queue_client_cache: OnceLock::new(),
            shared_queue: shared_tx,
            rx_of_shared_queue: Arc::new(tokio::sync::Mutex::new(rx_of_shared_queue)),
        };

        let real_base_controller = Arc::new(real_base_controller);
        // Start the background worker
        let controller_clone = real_base_controller.clone();
        rt.spawn(async move {
            controller_clone.bg_worker().await;
        });
        Ok(real_base_controller)
    }

    async fn bg_worker(&self) {
        const MIN_BACKOFF_ON_ERR: Duration = Duration::from_millis(100);
        const MAX_RETRIES: u32 = 5;

        debug!(
            "Launching core controller background task on thread {:?}",
            std::thread::current().name()
        );
        loop {
            // Receive actions from shared queue
            let mut rx = self.rx_of_shared_queue.lock().await;
            match rx.recv().await {
                Some(mut action) => {
                    let run_name = &action
                        .action_id
                        .run
                        .as_ref()
                        .map_or(String::from("<missing>"), |i| i.name.clone());
                    debug!(
                        "Controller worker processing action {}::{}",
                        run_name, action.action_id.name
                    );

                    // Drop the mutex guard before processing
                    drop(rx);

                    match self.handle_action(&mut action).await {
                        Ok(_) => {}
                        Err(e) => {
                            error!("Error in controller loop: {:?}", e);
                            // Handle backoff and retry logic
                            sleep(MIN_BACKOFF_ON_ERR).await;
                            action.retries += 1;

                            if action.retries > MAX_RETRIES {
                                error!(
                                    "Controller failed processing {}::{}, system retries {} crossed threshold {}",
                                    run_name, action.action_id.name, action.retries, MAX_RETRIES
                                );
                                action.client_err = Some(format!(
                                    "Controller failed {}::{}, system retries {} crossed threshold {}",
                                    run_name, action.action_id.name, action.retries, MAX_RETRIES
                                ));

                                // Fire completion event for failed action
                                if let Some(informer) = self.informer.get() {
                                    // todo: check these two errors

                                    // Before firing completion event, update the action in the
                                    // informer, otherwise client_err will not be set.
                                    let _ = informer.set_action_client_err(&action).await;
                                    let _ = informer
                                        .fire_completion_event(&action.action_id.name)
                                        .await;
                                } else {
                                    error!(
                                        "Max retries hit for action but informer still not yet initialized for action: {}",
                                        action.action_id.name
                                    );
                                }
                            } else {
                                // Re-queue the action for retry
                                info!(
                                    "Re-queuing action {}::{} for retry, attempt {}/{}",
                                    run_name, action.action_id.name, action.retries, MAX_RETRIES
                                );
                                if let Err(send_err) = self.shared_queue.send(action).await {
                                    error!("Failed to re-queue action for retry: {}", send_err);
                                }
                            }
                        }
                    }
                }
                None => {
                    warn!("Shared queue channel closed, stopping bg_worker");
                    break;
                }
            }
        }
    }

    async fn handle_action(&self, action: &mut Action) -> Result<(), ControllerError> {
        if !action.started {
            // Action not started, launch it
            self.bg_launch(action).await?;
        } else if action.is_action_terminal() {
            // Action is terminal, fire completion event
            if let Some(informer) = self.informer.get() {
                debug!(
                    "handle action firing completion event for {:?}",
                    &action.action_id.name
                );
                informer
                    .fire_completion_event(&action.action_id.name)
                    .await?;
            } else {
                error!(
                    "Informer not yet initialized for action: {}",
                    action.action_id.name
                );
                return Err(ControllerError::BadContext(format!(
                    "Informer not initialized for action: {}. This may be because the informer is still starting up.",
                    action.action_id.name
                )));
            }
        } else {
            // Action still in progress
            debug!("Resource {} still in progress...", action.action_id.name);
        }
        Ok(())
    }

    async fn bg_launch(&self, action: &Action) -> Result<(), ControllerError> {
        match self.launch_task(action).await {
            Ok(_) => {
                debug!("Successfully launched action: {}", action.action_id.name);
                Ok(())
            }
            Err(e) => {
                error!(
                    "Failed to launch action: {}, error: {}",
                    action.action_id.name, e
                );
                Err(ControllerError::RuntimeError(format!(
                    "Launch failed: {}",
                    e
                )))
            }
        }
    }

    async fn cancel_action(&self, action: &mut Action) -> Result<(), ControllerError> {
        if action.is_action_terminal() {
            info!(
                "Action {} is already terminal, no need to cancel.",
                action.action_id.name
            );
            return Ok(());
        }

        debug!("Cancelling action: {}", action.action_id.name);
        action.mark_cancelled();

        if let Some(informer) = self.informer.get() {
            let _ = informer
                .fire_completion_event(&action.action_id.name)
                .await?;
        } else {
            debug!(
                "Informer missing when trying to cancel action: {}",
                action.action_id.name
            );
        }
        Ok(())
    }

    async fn get_action(&self, action_id: ActionIdentifier) -> Result<Action, ControllerError> {
        if let Some(informer) = self.informer.get() {
            let action_name = action_id.name.clone();
            match informer.get_action(action_name).await {
                Some(action) => Ok(action),
                None => Err(ControllerError::RuntimeError(format!(
                    "Action not found: {}",
                    action_id.name
                ))),
            }
        } else {
            Err(ControllerError::BadContext(
                "Informer not initialized".to_string(),
            ))
        }
    }

    fn create_enqueue_action_request(
        &self,
        action: &Action,
    ) -> Result<EnqueueActionRequest, ControllerError> {
        // todo-pr: handle trace action
        let task_identifier = action
            .task
            .as_ref()
            .and_then(|task| task.task_template.as_ref())
            .and_then(|task_template| task_template.id.as_ref())
            .and_then(|core_task_id| {
                Some(TaskIdentifier {
                    version: core_task_id.version.clone(),
                    org: core_task_id.org.clone(),
                    project: core_task_id.project.clone(),
                    domain: core_task_id.domain.clone(),
                    name: core_task_id.name.clone(),
                })
            })
            .ok_or(ControllerError::RuntimeError(format!(
                "TaskIdentifier missing from Action {:?}",
                action
            )))?;

        let input_uri = action
            .inputs_uri
            .clone()
            .ok_or(ControllerError::RuntimeError(format!(
                "Inputs URI missing from Action {:?}",
                action
            )))?;
        let run_output_base =
            action
                .run_output_base
                .clone()
                .ok_or(ControllerError::RuntimeError(format!(
                    "Run output base missing from Action {:?}",
                    action
                )))?;
        let group = action.group.clone().unwrap_or_default();
        let task_action = TaskAction {
            id: Some(task_identifier),
            spec: action.task.clone(),
            cache_key: action
                .cache_key
                .as_ref()
                .map(|ck| StringValue { value: ck.clone() }),
            cluster: action.queue.clone().unwrap_or("".to_string()),
        };

        Ok(EnqueueActionRequest {
            action_id: Some(action.action_id.clone()),
            parent_action_name: Some(action.parent_action_name.clone()),
            spec: Some(enqueue_action_request::Spec::Task(task_action)),
            run_spec: None,
            input_uri,
            run_output_base,
            group,
            subject: String::default(), // Subject is not used in the current implementation
        })
    }

    async fn launch_task(&self, action: &Action) -> Result<EnqueueActionResponse, Status> {
        if !action.started && action.task.is_some() {
            let enqueue_request = self
                .create_enqueue_action_request(action)
                .expect("Failed to create EnqueueActionRequest");
            let mut client = self.queue_client();
            // todo: tonic doesn't seem to have wait_for_ready, or maybe the .ready is already doing this.
            let enqueue_result = client.enqueue_action(enqueue_request).await;
            match enqueue_result {
                Ok(response) => {
                    debug!("Successfully launched action: {:?}", action.action_id);
                    Ok(response.into_inner())
                }
                Err(e) => {
                    if e.code() == tonic::Code::AlreadyExists {
                        info!(
                            "Action {} already exists, continuing to monitor.",
                            action.action_id.name
                        );
                        Ok(EnqueueActionResponse {})
                    } else {
                        error!(
                            "Failed to launch action: {:?}, backing off...",
                            action.action_id
                        );
                        error!("Error details: {}", e);
                        // Handle backoff logic here
                        Err(e)
                    }
                }
            }
        } else {
            debug!(
                "Action {} is already started or has no task, skipping launch.",
                action.action_id.name
            );
            Ok(EnqueueActionResponse {})
        }
    }

    pub async fn _submit_action(&self, action: Action) -> Result<Action, ControllerError> {
        let action_name = action.action_id.name.clone();
        let parent_action_name = action.parent_action_name.clone();
        // The first action that gets submitted determines the run_id that will be used.
        // This is obviously not going to work,

        let run_id = action
            .action_id
            .run
            .clone()
            .ok_or(ControllerError::RuntimeError(format!(
                "Run ID missing from submit action {}",
                action_name.clone()
            )))?;
        let informer: &Arc<Informer> = self
            .informer // OnceCell<Arc<Informer>>
            .get_or_try_init(|| async move {
                info!("Creating informer set to run_id {:?}", run_id);
                let inf = Arc::new(Informer::new(
                    self.state_client(),
                    run_id,
                    parent_action_name,
                    self.shared_queue.clone(),
                ));

                Informer::start(inf.clone()).await?;

                // Using PyErr for now, but any errors coming from the informer will not really
                // be py errs, will need to add and map later.
                Ok::<Arc<Informer>, ControllerError>(inf)
            })
            .await?;
        let (done_tx, done_rx) = oneshot::channel();
        informer.submit_action(action, done_tx).await?;

        done_rx.await.map_err(|_| {
            ControllerError::BadContext(String::from("Failed to receive done signal from informer"))
        })?;
        debug!(
            "Action {} complete, looking up final value and returning",
            action_name
        );

        // get the action and return it
        let final_action = informer.get_action(action_name).await;
        final_action.ok_or(ControllerError::BadContext(String::from(
            "Action not found after done",
        )))
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
            use crate::auth::{AuthConfig, AuthLayer, ClientCredentialsAuthenticator};
            use flyteidl2::flyteidl::common::ListRequest;
            use flyteidl2::flyteidl::common::ProjectIdentifier;
            use flyteidl2::flyteidl::task::task_service_client::TaskServiceClient;
            use flyteidl2::flyteidl::task::{list_tasks_request, ListTasksRequest};
            use tonic::Code;
            use tower::ServiceBuilder;

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
            let endpoint = create_tls_endpoint(static_endpoint, domain).await?;
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
            use crate::auth::{AuthConfig, AuthLayer, ClientCredentialsAuthenticator};
            use flyteidl2::flyteidl::common::ActionIdentifier;
            use flyteidl2::flyteidl::common::RunIdentifier;
            use flyteidl2::flyteidl::workflow::watch_request::Filter;
            use flyteidl2::flyteidl::workflow::WatchRequest;
            use std::time::Duration;
            use tokio::time::sleep;
            use tower::ServiceBuilder;

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
            let endpoint = create_tls_endpoint(static_endpoint, domain).await?;
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
            real_base._submit_action(action).await.map_err(|e| {
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
fn flyte_controller_base(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
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
