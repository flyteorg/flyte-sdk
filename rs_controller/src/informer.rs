use crate::action::Action;
use crate::core::{ControllerError, StateClient};

use flyteidl2::flyteidl::common::ActionIdentifier;
use flyteidl2::flyteidl::common::RunIdentifier;
use flyteidl2::flyteidl::workflow::state_service_client::StateServiceClient;
use flyteidl2::flyteidl::workflow::{
    watch_request, watch_response::Message, WatchRequest, WatchResponse,
};

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::select;
use tokio::sync::RwLock;
use tokio::sync::{mpsc, oneshot, Notify};
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tonic::transport::channel::Channel;
use tonic::transport::Endpoint;
use tracing::{debug, error, info, warn};
use tracing_subscriber::fmt;

#[derive(Clone, Debug)]
pub struct Informer {
    client: StateClient,
    run_id: RunIdentifier,
    action_cache: Arc<RwLock<HashMap<String, Action>>>,
    parent_action_name: String,
    shared_queue: mpsc::Sender<Action>,
    ready: Arc<Notify>,
    completion_events: Arc<RwLock<HashMap<String, oneshot::Sender<()>>>>,
}

impl Informer {
    pub fn new(
        client: StateClient,
        run_id: RunIdentifier,
        parent_action_name: String,
        shared_queue: mpsc::Sender<Action>,
    ) -> Self {
        Informer {
            client,
            run_id,
            action_cache: Arc::new(RwLock::new(HashMap::new())),
            parent_action_name,
            shared_queue,
            ready: Arc::new(Notify::new()),
            completion_events: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn set_action_client_err(&self, action: &Action) -> Result<(), ControllerError> {
        if let Some(client_err) = &action.client_err {
            let mut cache = self.action_cache.write().await;
            let action_name = action.action_id.name.clone();
            if let Some(action) = cache.get_mut(&action_name) {
                action.set_client_err(client_err.clone());
                Ok(())
            } else {
                Err(ControllerError::RuntimeError(format!(
                    "Action {} not found in cache",
                    action_name
                )))
            }
        } else {
            Ok(())
        }
    }

    async fn handle_watch_response(
        &self,
        response: WatchResponse,
    ) -> Result<Option<Action>, ControllerError> {
        debug!(
            "Informer for {:?}::{} processing incoming message {:?}",
            self.run_id.name, self.parent_action_name, &response
        );
        if let Some(msg) = response.message {
            match msg {
                Message::ControlMessage(_) => {
                    // Handle control messages if needed
                    debug!("Received sentinel for parent {}", self.parent_action_name);
                    self.ready.notify_one();
                    Ok(None)
                }
                Message::ActionUpdate(action_update) => {
                    // Handle action updates
                    debug!("Received action update: {:?}", action_update.action_id);
                    let mut cache = self.action_cache.write().await;
                    let action_name = action_update
                        .action_id
                        .as_ref()
                        .map(|act_id| act_id.name.clone())
                        .ok_or(ControllerError::RuntimeError(format!(
                            "Action update received without a name: {:?}",
                            action_update
                        )))?;

                    if let Some(existing) = cache.get_mut(&action_name) {
                        existing.merge_update(&action_update);

                        // Don't fire a completion event here either - successful return of this
                        // function should re-enqueue the action for processing, and the controller
                        // will detect and fire completion
                    } else {
                        debug!(
                            "Action update for {:?} not in cache, adding",
                            action_update.action_id
                        );
                        let action_from_update =
                            Action::new_from_update(self.parent_action_name.clone(), action_update);
                        cache.insert(action_name.clone(), action_from_update);

                        // don't fire completion events here because we may not have a completion event yet
                        // i.e. the submit that creates the completion event may not have fired yet, so just
                        // add to the cache for now.
                    }

                    Ok(Some(cache.get(&action_name).unwrap().clone()))
                }
            }
        } else {
            Err(ControllerError::BadContext(
                "No message in response".to_string(),
            ))
        }
    }

    async fn watch_actions(&self) -> ControllerError {
        let action_id = ActionIdentifier {
            name: self.parent_action_name.clone(),
            run: Some(self.run_id.clone()),
        };
        let request = WatchRequest {
            filter: Some(watch_request::Filter::ParentActionId(action_id)),
        };

        let mut stream = self.client.clone().watch(request).await;

        let mut stream = match stream {
            Ok(s) => s.into_inner(),
            Err(e) => {
                error!("Failed to start watch stream: {:?}", e);
                return ControllerError::from(e);
            }
        };

        loop {
            match stream.message().await {
                Ok(Some(response)) => {
                    let handle_response = self.handle_watch_response(response).await;
                    match handle_response {
                        Ok(Some(action)) => match self.shared_queue.send(action).await {
                            Ok(_) => {
                                continue;
                            }
                            Err(e) => {
                                error!("Informer watch failed sending action back to shared queue: {:?}", e);
                                return ControllerError::RuntimeError(format!(
                                    "Failed to send action to shared queue: {}",
                                    e
                                ));
                            }
                        },
                        Ok(None) => {
                            debug!(
                                "Received None from handle_watch_response, continuing watch loop."
                            );
                        }
                        Err(err) => {
                            // this should cascade up to the controller to restart the informer, and if there
                            // are too many informer restarts, the controller should fail
                            error!("Error in informer watch {:?}", err);
                            return err;
                        }
                    }
                }
                Ok(None) => {
                    debug!("Stream received empty message, maybe no more messages? Repeating watch loop.");
                } // Stream ended, exit loop
                Err(e) => {
                    error!("Error receiving message from stream: {:?}", e);
                    return ControllerError::from(e);
                }
            }
        }
    }

    async fn wait_ready_or_timeout(ready: Arc<Notify>) -> Result<(), ControllerError> {
        select! {
            _ = ready.notified() => {
                debug!("Ready sentinel ack'ed");
                Ok(())
            }
            _ = sleep(Duration::from_millis(100)) => Err(ControllerError::SystemError("".to_string()))
        }
    }

    pub async fn start(informer: Arc<Self>) -> Result<JoinHandle<()>, ControllerError> {
        let me = informer.clone();
        let ready = me.ready.clone();
        let _watch_handle = tokio::spawn(async move {
            // handle errors later
            me.watch_actions().await;
        });

        match Self::wait_ready_or_timeout(ready).await {
            Ok(()) => Ok(_watch_handle),
            Err(_) => {
                warn!("Timed out waiting for sentinel");
                Ok(_watch_handle)
            }
        }
    }

    pub async fn get_action(&self, action_name: String) -> Option<Action> {
        let cache = self.action_cache.read().await;
        cache.get(&action_name).cloned()
    }

    pub async fn submit_action(
        &self,
        action: Action,
        done_tx: oneshot::Sender<()>,
    ) -> Result<(), ControllerError> {
        let action_name = action.action_id.name.clone();

        // Store the completion event sender
        {
            let mut completion_events = self.completion_events.write().await;
            completion_events.insert(action_name.clone(), done_tx);
        }

        // Add action to shared queue
        self.shared_queue.send(action).await.map_err(|e| {
            ControllerError::RuntimeError(format!("Failed to send action to shared queue: {}", e))
        })?;

        Ok(())
    }

    pub async fn fire_completion_event(&self, action_name: &str) -> Result<(), ControllerError> {
        info!("Firing completion event for action: {}", action_name);
        let mut completion_events = self.completion_events.write().await;
        if let Some(done_tx) = completion_events.remove(action_name) {
            done_tx.send(()).map_err(|_| {
                ControllerError::RuntimeError(format!(
                    "Failed to send completion event for action: {}",
                    action_name
                ))
            })?;
        } else {
            error!(
                "No completion event found for action---------------------: {}",
                action_name,
            );
            // Return error, which should cause informer to re-enqueue
            return Err(ControllerError::RuntimeError(format!(
                "No completion event found for action: {}. This may be because the informer is still starting up.",
                action_name
            )));
        }
        Ok(())
    }
}

async fn informer_main() {
    // Create an informer but first create the shared_queue that will be shared between the
    // Controller and the informer
    let (tx, rx) = mpsc::channel::<Action>(64);
    let endpoint = Endpoint::from_static("http://localhost:8090");
    let channel = endpoint.connect().await.unwrap();
    let client = StateServiceClient::new(channel);

    let run_id = RunIdentifier {
        org: String::from("testorg"),
        project: String::from("testproject"),
        domain: String::from("development"),
        name: String::from("qdtc266r2z8clscl2lj5"),
    };

    let informer = Arc::new(Informer::new(
        StateClient::Plain(client),
        run_id,
        "a0".to_string(),
        tx.clone(),
    ));

    let watch_task = Informer::start(informer.clone()).await;

    println!("{:?}: {:?}", informer, watch_task);
    // do creation and start of informer behind a once
}

fn init_tracing() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let subscriber = fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer() // so logs show in test output
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("setting default subscriber failed");
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_informer() {
        init_tracing();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(informer_main());
    }
}
