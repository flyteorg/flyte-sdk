mod action;
mod informer;

use std::sync::Arc;
use std::time::Duration;

use futures::TryFutureExt;
use pyo3::prelude::*;
use tokio::sync::{mpsc, Mutex, Notify};
use tracing::{debug, error, info, warn};

use thiserror::Error;

use crate::action::{Action, ActionType};
use crate::informer::Informer;

use cloudidl::cloudidl::workflow::queue_service_client::QueueServiceClient;
use cloudidl::cloudidl::workflow::state_service_client::StateServiceClient;
use cloudidl::cloudidl::workflow::{
    enqueue_action_request, ActionIdentifier, EnqueueActionRequest, EnqueueActionResponse,
    RunIdentifier, TaskAction, TaskIdentifier,
};
use cloudidl::google;
use google::protobuf::StringValue;
use pyo3::exceptions;
use pyo3::types::PyAny;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_async_runtimes::tokio::get_runtime;
use tokio::sync::{oneshot, OnceCell};
use tokio::time::sleep;
use tonic::transport::Endpoint;
use tonic::Status;
use tracing_subscriber::FmtSubscriber;

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

struct CoreBaseController {
    endpoint: String,
    state_client: StateServiceClient<tonic::transport::Channel>,
    queue_client: QueueServiceClient<tonic::transport::Channel>,
    informer: OnceCell<Arc<Informer>>,
    shared_queue: mpsc::Sender<Action>,
    rx_of_shared_queue: Arc<tokio::sync::Mutex<mpsc::Receiver<Action>>>,
}

impl CoreBaseController {
    pub fn try_new(endpoint: String) -> Result<Arc<Self>, ControllerError> {
        info!("Creating CoreBaseController with endpoint {:?}", endpoint);
        // play with taking str slice instead of String instead of intentionally leaking.
        let endpoint_static: &'static str = Box::leak(Box::new(endpoint.clone().into_boxed_str()));
        // shared queue
        let (shared_tx, rx_of_shared_queue) = mpsc::channel::<Action>(64);

        let rt = get_runtime();
        let (state_client, queue_client) = rt.block_on(async {
            // Need to update to with auth to read API key
            let endpoint = Endpoint::from_static(&endpoint_static);
            let channel = endpoint.connect().await.map_err(|e| ControllerError::from(e))?;
            Ok::<_, ControllerError>((
                StateServiceClient::new(channel.clone()),
                QueueServiceClient::new(channel),
            ))
        })?;

        let real_base_controller = CoreBaseController {
            endpoint,
            state_client,
            queue_client,
            informer: OnceCell::new(),
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
                    self.state_client.clone(),
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
    #[pyo3(signature = (*, endpoint))]
    fn new(endpoint: String) -> PyResult<Self> {
        info!("Creating controller wrapper with endpoint {:?}", endpoint);
        let core_base = CoreBaseController::try_new(endpoint)?;
        Ok(BaseController(core_base))
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

use cloudidl::pymodules::cloud_mod;

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
    cloud_mod(py, m)?;
    Ok(())
}
