//! Core controller implementation - Pure Rust, no PyO3 dependencies
//! This module can be used by both Python bindings and standalone Rust binaries

use std::sync::Arc;
use std::time::Duration;

use pyo3_async_runtimes::tokio::get_runtime;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::time::sleep;
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint};
use tonic::Status;
use tower::ServiceBuilder;
use tracing::{debug, error, info, warn};

use crate::action::Action;
use crate::auth::{AuthConfig, AuthLayer, ClientCredentialsAuthenticator};
use crate::error::{ControllerError, InformerError};
use crate::informer::{Informer, InformerCache};
use flyteidl2::flyteidl::common::{ActionIdentifier, RunIdentifier};
use flyteidl2::flyteidl::task::TaskIdentifier;
use flyteidl2::flyteidl::workflow::enqueue_action_request;
use flyteidl2::flyteidl::workflow::queue_service_client::QueueServiceClient;
use flyteidl2::flyteidl::workflow::state_service_client::StateServiceClient;
use flyteidl2::flyteidl::workflow::{
    EnqueueActionRequest, EnqueueActionResponse, TaskAction, WatchRequest, WatchResponse,
};
use flyteidl2::google;
use google::protobuf::StringValue;

// Fetches Amazon root CA certificate from Amazon Trust Services
pub async fn fetch_amazon_root_ca() -> Result<Certificate, ControllerError> {
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
pub async fn create_tls_endpoint(
    url: &'static str,
    domain: &str,
) -> Result<Endpoint, ControllerError> {
    // Fetch Amazon root CA dynamically
    let cert = fetch_amazon_root_ca().await?;

    let tls_config = ClientTlsConfig::new()
        .domain_name(domain)
        .ca_certificate(cert);

    let endpoint = Endpoint::from_static(url)
        .tls_config(tls_config)
        .map_err(|e| ControllerError::SystemError(format!("TLS config error: {}", e)))?
        .keep_alive_while_idle(true);

    Ok(endpoint)
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

pub struct CoreBaseController {
    informer_cache: InformerCache,
    queue_client: QueueClient,
    shared_queue: mpsc::Sender<Action>,
    shared_queue_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<Action>>>,
    failure_rx: Arc<std::sync::Mutex<Option<mpsc::Receiver<InformerError>>>>,
    bg_worker_handle: Arc<std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl CoreBaseController {
    pub fn new_with_auth() -> Result<Arc<Self>, ControllerError> {
        info!("Creating CoreBaseController from _UNION_EAGER_API_KEY env var (with auth)");
        // Read from env var and use auth
        let api_key = std::env::var("_UNION_EAGER_API_KEY").map_err(|_| {
            ControllerError::SystemError(
                "_UNION_EAGER_API_KEY env var must be provided".to_string(),
            )
        })?;
        let auth_config = AuthConfig::new_from_api_key(&api_key)?;
        let endpoint_url = auth_config.endpoint.clone();

        let endpoint_static: &'static str =
            Box::leak(Box::new(endpoint_url.clone().into_boxed_str()));
        // shared queue
        let (shared_tx, shared_queue_rx) = mpsc::channel::<Action>(64);

        let rt = get_runtime();
        let channel = rt.block_on(async {
            // todo: escape hatch for localhost
            // Strip "https://" to get just the hostname for TLS config
            let domain = endpoint_url.strip_prefix("https://").ok_or_else(|| {
                ControllerError::SystemError(
                    "Endpoint must start with https:// when using auth".to_string(),
                )
            })?;

            // Create TLS-configured endpoint
            let endpoint = create_tls_endpoint(endpoint_static, domain).await?;
            let channel = endpoint.connect().await.map_err(ControllerError::from)?;

            let authenticator = Arc::new(ClientCredentialsAuthenticator::new(auth_config.clone()));
            let auth_channel = ServiceBuilder::new()
                .layer(AuthLayer::new(authenticator, channel.clone()))
                .service(channel);

            Ok::<_, ControllerError>(ChannelType::Authenticated(auth_channel))
        })?;

        let (failure_tx, failure_rx) = mpsc::channel::<InformerError>(10);

        let state_client = match &channel {
            ChannelType::Plain(ch) => StateClient::Plain(StateServiceClient::new(ch.clone())),
            ChannelType::Authenticated(ch) => {
                StateClient::Authenticated(StateServiceClient::new(ch.clone()))
            }
        };

        let queue_client = match &channel {
            ChannelType::Plain(ch) => QueueClient::Plain(QueueServiceClient::new(ch.clone())),
            ChannelType::Authenticated(ch) => {
                QueueClient::Authenticated(QueueServiceClient::new(ch.clone()))
            }
        };

        let informer_cache =
            InformerCache::new(state_client.clone(), shared_tx.clone(), failure_tx);

        let real_base_controller = CoreBaseController {
            informer_cache,
            queue_client,
            shared_queue: shared_tx,
            shared_queue_rx: Arc::new(tokio::sync::Mutex::new(shared_queue_rx)),
            failure_rx: Arc::new(std::sync::Mutex::new(Some(failure_rx))),
            bg_worker_handle: Arc::new(std::sync::Mutex::new(None)),
        };

        let real_base_controller = Arc::new(real_base_controller);
        // Start the background worker
        let controller_clone = real_base_controller.clone();
        let handle = rt.spawn(async move {
            controller_clone.bg_worker().await;
        });

        // Store the handle
        *real_base_controller.bg_worker_handle.lock().unwrap() = Some(handle);

        Ok(real_base_controller)
    }

    pub fn new_without_auth(endpoint: String) -> Result<Arc<Self>, ControllerError> {
        let endpoint_static: &'static str = Box::leak(Box::new(endpoint.clone().into_boxed_str()));
        // shared queue
        let (shared_tx, shared_queue_rx) = mpsc::channel::<Action>(64);

        let rt = get_runtime();
        let channel = rt.block_on(async {
            let chan = if endpoint.starts_with("http://") {
                let endpoint = Endpoint::from_static(endpoint_static).keep_alive_while_idle(true);
                endpoint.connect().await.map_err(ControllerError::from)?
            } else if endpoint.starts_with("https://") {
                // Strip "https://" to get just the hostname for TLS config
                let domain = endpoint.strip_prefix("https://").ok_or_else(|| {
                    ControllerError::SystemError("Endpoint must start with https://".to_string())
                })?;

                // Create TLS-configured endpoint
                let endpoint = create_tls_endpoint(endpoint_static, domain).await?;
                endpoint.connect().await.map_err(ControllerError::from)?
            } else {
                return Err(ControllerError::SystemError(format!(
                    "Malformed endpoint {}",
                    endpoint
                )));
            };
            Ok::<_, ControllerError>(ChannelType::Plain(chan))
        })?;

        let (failure_tx, failure_rx) = mpsc::channel::<InformerError>(10);

        let state_client = match &channel {
            ChannelType::Plain(ch) => StateClient::Plain(StateServiceClient::new(ch.clone())),
            ChannelType::Authenticated(ch) => {
                StateClient::Authenticated(StateServiceClient::new(ch.clone()))
            }
        };

        let queue_client = match &channel {
            ChannelType::Plain(ch) => QueueClient::Plain(QueueServiceClient::new(ch.clone())),
            ChannelType::Authenticated(ch) => {
                QueueClient::Authenticated(QueueServiceClient::new(ch.clone()))
            }
        };

        let informer_cache =
            InformerCache::new(state_client.clone(), shared_tx.clone(), failure_tx);

        let real_base_controller = CoreBaseController {
            informer_cache,
            queue_client,
            shared_queue: shared_tx,
            shared_queue_rx: Arc::new(tokio::sync::Mutex::new(shared_queue_rx)),
            failure_rx: Arc::new(std::sync::Mutex::new(Some(failure_rx))),
            bg_worker_handle: Arc::new(std::sync::Mutex::new(None)),
        };

        let real_base_controller = Arc::new(real_base_controller);
        // Start the background worker
        let controller_clone = real_base_controller.clone();
        let handle = rt.spawn(async move {
            controller_clone.bg_worker().await;
        });

        // Store the handle
        *real_base_controller.bg_worker_handle.lock().unwrap() = Some(handle);

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
            let mut rx = self.shared_queue_rx.lock().await;
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
                        // Add handling here for new slow down error
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
                                let opt_informer = self
                                    .informer_cache
                                    .get(&action.get_run_identifier(), &action.parent_action_name)
                                    .await;
                                if let Some(informer) = opt_informer {
                                    // Before firing completion event, update the action in the
                                    // informer, otherwise client_err will not be set.
                                    // todo: gain a better understanding of these two errors and handle
                                    let res = informer.set_action_client_err(&action).await;
                                    match res {
                                        Ok(()) => {}
                                        Err(e) => {
                                            error!(
                                                "Error setting error for failed action {}: {}",
                                                &action.get_full_name(),
                                                e
                                            )
                                        }
                                    }
                                    let res = informer
                                        .fire_completion_event(&action.action_id.name)
                                        .await;
                                    match res {
                                        Ok(()) => {}
                                        Err(e) => {
                                            error!("Error firing completion event for failed action {}: {}", &action.get_full_name(), e)
                                        }
                                    }
                                } else {
                                    error!(
                                        "Max retries hit for action but informer missing: {:?}",
                                        action.action_id
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
            warn!("Action is not started, launching action {:?}", action);
            self.bg_launch(action).await?;
        } else if action.is_action_terminal() {
            // Action is terminal, fire completion event
            if let Some(arc_informer) = self
                .informer_cache
                .get(&action.get_run_identifier(), &action.parent_action_name)
                .await
            {
                debug!(
                    "handle action firing completion event for {:?}",
                    &action.action_id.name
                );
                arc_informer
                    .fire_completion_event(&action.action_id.name)
                    .await?;
            } else {
                error!(
                    "Unable to find informer to fire completion event for action: {}",
                    action.get_full_name(),
                );
                return Err(ControllerError::BadContext(format!(
                    "Informer missing for action: {} while handling.",
                    action.get_full_name()
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

    pub async fn cancel_action(&self, action: &mut Action) -> Result<(), ControllerError> {
        if action.is_action_terminal() {
            info!(
                "Action {} is already terminal, no need to cancel.",
                action.action_id.name
            );
            return Ok(());
        }

        // debug
        warn!("Cancelling action!!!: {}", action.action_id.name);
        action.mark_cancelled();

        if let Some(informer) = self
            .informer_cache
            .get(&action.get_run_identifier(), &action.parent_action_name)
            .await
        {
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

    pub async fn get_action(
        &self,
        action_id: ActionIdentifier,
        parent_action_name: &str,
    ) -> Result<Action, ControllerError> {
        let run = action_id
            .run
            .as_ref()
            .ok_or(ControllerError::RuntimeError(format!(
                "Action {:?} doesn't have a run, can't get action",
                action_id
            )))?;
        if let Some(informer) = self.informer_cache.get(run, parent_action_name).await {
            let action_name = action_id.name.clone();
            match informer.get_action(action_name).await {
                Some(action) => Ok(action),
                None => Err(ControllerError::RuntimeError(format!(
                    "Action not found getting from action_id: {:?}",
                    action_id
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
            let mut client = self.queue_client.clone();
            // todo: tonic doesn't seem to have wait_for_ready, or maybe the .ready is already doing this.
            let enqueue_result = client.enqueue_action(enqueue_request).await;
            // Add logic from resiliency pr here, return certain errors, but change others to be a specific slowdown error.
            match enqueue_result {
                Ok(response) => {
                    debug!("Successfully enqueued action: {:?}", action.action_id);
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

    pub async fn submit_action(&self, action: Action) -> Result<Action, ControllerError> {
        let action_name = action.action_id.name.clone();
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
        info!("Creating informer set to run_id {:?}", run_id);
        let informer: Arc<Informer> = self
            .informer_cache
            .get_or_create_informer(&action.get_run_identifier(), &action.parent_action_name)
            .await;
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

    pub async fn finalize_parent_action(&self, run_id: &RunIdentifier, parent_action_name: &str) {
        let opt_informer = self.informer_cache.remove(run_id, parent_action_name).await;
        match opt_informer {
            Some(informer) => {
                informer.stop().await;
            }
            None => {
                warn!(
                    "No informer found when finalizing parent action {}",
                    parent_action_name
                );
            }
        }
    }

    pub async fn watch_for_errors(&self) -> Result<(), ControllerError> {
        // Take ownership of both (can only be called once)
        let handle = self.bg_worker_handle.lock().unwrap().take();
        let failure_rx = self.failure_rx.lock().unwrap().take();

        match (handle, failure_rx) {
            (Some(handle), Some(mut rx)) => {
                // Race bg_worker completion vs informer errors
                tokio::select! {
                    // bg_worker completed or panicked
                    result = handle => {
                        match result {
                            Ok(_) => {
                                error!("Background worker exited unexpectedly");
                                Err(ControllerError::RuntimeError(
                                    "Background worker exited unexpectedly".to_string(),
                                ))
                            }
                            Err(e) if e.is_panic() => {
                                error!("Background worker panicked: {:?}", e);
                                Err(ControllerError::RuntimeError(format!(
                                    "Background worker panicked: {:?}",
                                    e
                                )))
                            }
                            Err(e) => {
                                error!("Background worker was cancelled: {:?}", e);
                                Err(ControllerError::RuntimeError(format!(
                                    "Background worker cancelled: {:?}",
                                    e
                                )))
                            }
                        }
                    }

                    // Informer error received
                    informer_err = rx.recv() => {
                        match informer_err {
                            Some(err) => {
                                error!("Informer error received: {:?}", err);
                                Err(ControllerError::Informer(err))
                            }
                            None => {
                                error!("Informer error channel closed unexpectedly");
                                Err(ControllerError::RuntimeError(
                                    "Informer error channel closed unexpectedly".to_string(),
                                ))
                            }
                        }
                    }
                }
            }
            _ => Err(ControllerError::RuntimeError(
                "watch_for_errors already called or resources not available".to_string(),
            )),
        }
    }
}
