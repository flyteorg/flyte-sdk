use crate::action::Action;
use crate::core::StateClient;
use crate::error::{ControllerError, InformerError};

use flyteidl2::flyteidl::common::ActionIdentifier;
use flyteidl2::flyteidl::common::RunIdentifier;
use flyteidl2::flyteidl::workflow::state_service_client::StateServiceClient;
use flyteidl2::flyteidl::workflow::{
    watch_request, watch_response::Message, WatchRequest, WatchResponse,
};

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use pyo3_async_runtimes::tokio::run;
use tokio::select;
use tokio::sync::RwLock;
use tokio::sync::{mpsc, oneshot, Notify};
use tokio::time::sleep;
use tonic::transport::channel::Channel;
use tonic::transport::Endpoint;
use tracing::{debug, error, info, warn};
use tracing::log::Level::Info;
use tracing_subscriber::fmt;

#[derive(Clone, Debug)]
pub struct Informer {
    client: StateClient,
    run_id: RunIdentifier,
    action_cache: Arc<RwLock<HashMap<String, Action>>>,
    parent_action_name: String,
    shared_queue: mpsc::Sender<Action>,
    ready: Arc<Notify>,
    is_ready: Arc<AtomicBool>,
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
            is_ready: Arc::new(AtomicBool::new(false)),
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
                    self.is_ready.store(true, Ordering::Release);
                    self.ready.notify_waiters();
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

        let stream = self.client.clone().watch(request).await;

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

pub struct InformerCache {
    cache: Arc<RwLock<HashMap<String, Arc<Informer>>>>,
    client: StateClient,
    shared_queue: mpsc::Sender<Action>,
    failure_tx: mpsc::Sender<InformerError>,
}

impl InformerCache {
    pub fn new(
        client: StateClient,
        shared_queue: mpsc::Sender<Action>,
        failure_tx: mpsc::Sender<InformerError>,
    ) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            client,
            shared_queue,
            failure_tx,
        }
    }

    fn mkname(run_name: &str, parent_action_name: &str) -> String {
        format!("{}.{}", run_name, parent_action_name)
    }

    pub async fn get_or_create_informer(
        &self,
        run_id: &RunIdentifier,
        parent_action_name: &str,
    ) -> Arc<Informer> {
        let informer_name = Self::mkname(&run_id.name, parent_action_name);
        let timeout = Duration::from_millis(100);

        // Check if exists (with read lock)
        {
            let map = self.cache.read().await;
            if let Some(informer) = map.get(&informer_name) {
                let arc_informer = Arc::clone(informer);
                // Release read lock before waiting
                drop(map);
                Self::wait_for_ready(&arc_informer, timeout).await;
                return arc_informer;
            }
        }

        // Create new informer (with write lock)
        let mut map = self.cache.write().await;

        // Double-check it wasn't created while we were waiting for write lock
        if let Some(informer) = map.get(&informer_name) {
            let arc_informer = Arc::clone(informer);
            drop(map);
            Self::wait_for_ready(&arc_informer, timeout).await;
            return arc_informer;
        }

        // Create and add to cache
        let informer = Arc::new(Informer::new(
            self.client.clone(),
            run_id.clone(),
            parent_action_name.to_string(),
            self.shared_queue.clone(),
        ));
        map.insert(informer_name.clone(), Arc::clone(&informer));

        // Release write lock before starting (starting involves waiting)
        drop(map);

        let me = Arc::clone(&informer);
        let failure_tx = self.failure_tx.clone();

        // todo: Add stop and terminate watch handle
        let _watch_handle = tokio::spawn(async move {
            // If there are errors with the watch then notify the channel
            let err = me.watch_actions().await;

            error!(
                "Informer watch_actions failed for run {}, parent action {}: {:?}",
                me.run_id.name, me.parent_action_name, err
            );

            let failure = InformerError::WatchFailed {
                run_name: me.run_id.name.clone(),
                parent_action_name: me.parent_action_name.clone(),
                error_message: err.to_string(),
            };

            if let Err(e) = failure_tx.send(failure).await {
                error!("Failed to send informer failure event: {:?}", e);
            }
        });

        // Optimistically wait for ready (sentinel) with timeout
        Self::wait_for_ready(&informer, timeout).await;

        informer
    }

    pub async fn get(&self, run_id: &RunIdentifier,
                 parent_action_name: &str) -> Option<Arc<Informer>> {
        let map = self.cache.read().await;
        let opt_informer = map.get(&InformerCache::mkname(&run_id.name, parent_action_name)).cloned();
        opt_informer
    }

    /// Wait for informer to be ready with a timeout. If timeout occurs, set ready anyway
    /// and log a warning - this is optimistic, assuming the informer will become ready eventually.
    /// Once ready has been set, future calls return immediately without waiting.
    async fn wait_for_ready(informer: &Arc<Informer>, timeout: Duration) {
        // Subscribe to notifications first, before checking ready
        // This ensures we don't miss a notification that happens between the check and the wait
        let ready_fut = informer.ready.notified();

        // Quick check - if already ready, return immediately
        if informer.is_ready.load(Ordering::Acquire) {
            debug!(
                "Informer already ready for parent_action: {}",
                informer.parent_action_name
            );
            return;
        }

        // Otherwise wait with timeout
        match tokio::time::timeout(timeout, ready_fut).await {
            Ok(_) => {
                info!(
                    "Informer ready for parent_action: {}",
                    informer.parent_action_name
                );
            }
            Err(_) => {
                warn!(
                    "Informer cache sync timed out after {:?} for {}:{} - continuing optimistically",
                    timeout, informer.run_id.name, informer.parent_action_name
                );
                // Set ready anyway so future calls don't wait
                informer.is_ready.store(true, Ordering::Release);
            }
        }
    }
}

async fn informer_main() {
    // Create an informer but first create the shared_queue that will be shared between the
    // Controller and the informer
    let (tx, _rx) = mpsc::channel::<Action>(64);
    let endpoint = Endpoint::from_static("http://localhost:8090");
    let channel = endpoint.connect().await.unwrap();
    let client = StateServiceClient::new(channel);

    let run_id = RunIdentifier {
        org: String::from("testorg"),
        project: String::from("testproject"),
        domain: String::from("development"),
        name: String::from("qdtc266r2z8clscl2lj5"),
    };
    let (failure_tx, _failure_rx) = mpsc::channel::<InformerError>(1);

    let informer_cache = InformerCache::new(StateClient::Plain(client), tx.clone(), failure_tx);
    let informer = informer_cache
        .get_or_create_informer(&run_id, "a0")
        .await;

    println!("{:?}", informer);
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
