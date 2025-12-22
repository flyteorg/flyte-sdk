use crate::action::Action;
use crate::core::StateClient;
use crate::error::{ControllerError, InformerError};
use tokio::time;
use tokio_util::sync::CancellationToken;
/// Determine if an InformerError is retryable
fn is_retryable_error(err: &InformerError) -> bool {
    match err {
        // Retryable gRPC and stream errors
        InformerError::GrpcError(_) => true,
        InformerError::StreamError(_) => true,

        // Don't retry these
        InformerError::Cancelled => false,
        InformerError::BadContext(_) => false,
        InformerError::QueueSendError(_) => false,
        InformerError::WatchFailed { .. } => false,
    }
}

use flyteidl2::flyteidl::common::ActionIdentifier;
use flyteidl2::flyteidl::common::RunIdentifier;
use flyteidl2::flyteidl::workflow::{
    watch_request, watch_response::Message, WatchRequest, WatchResponse,
};

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::select;
use tokio::sync::RwLock;
use tokio::sync::{mpsc, oneshot, Notify};
use tracing::{debug, error, info, warn};

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
    cancellation_token: CancellationToken,
    watch_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
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
            cancellation_token: CancellationToken::new(),
            watch_handle: Arc::new(RwLock::new(None)),
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
    ) -> Result<Option<Action>, InformerError> {
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
                        .ok_or(InformerError::StreamError(format!(
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
            Err(InformerError::BadContext(
                "No message in response".to_string(),
            ))
        }
    }

    async fn watch_actions(&self) -> Result<(), InformerError> {
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
                return Err(InformerError::from(e));
            }
        };

        loop {
            select! {
                _ = self.cancellation_token.cancelled() => {
                    warn!("Cancellation token got - exiting from watch_actions: {}", self.parent_action_name);
                    return Err(InformerError::Cancelled)
                }

                result = stream.message() => {
                    match result {
                        Ok(Some(response)) => {
                            let handle_response = self.handle_watch_response(response).await;
                            match handle_response {
                                Ok(Some(action)) => match self.shared_queue.send(action).await {
                                    Ok(_) => {
                                        // continue;
                                        // MOCK: Simulate failure for testing after put
                                        // terminal actor to channel
                                        error!("Simulating error!!! This is an error!!!");
                                        return Err(InformerError::BadContext("Simulated failure for testing".to_string()));
                                    }
                                    Err(e) => {
                                        error!("Informer watch failed sending action back to shared queue: {:?}", e);
                                        return Err(InformerError::QueueSendError(format!(
                                            "Failed to send action to shared queue: {}",
                                            e
                                        )));
                                    }
                                },
                                Ok(None) => {
                                    debug!(
                                        "Received None from handle_watch_response, continuing watch loop."
                                    );
                                }
                                Err(err) => {
                                    // this should cascade up to retry logic
                                    error!("Error in informer watch {:?}", err);
                                    return Err(err);
                                }
                            }
                        }
                        Ok(None) => {
                            debug!("Stream received empty message, maybe no more messages? Repeating watch loop.");
                        } // Stream ended, exit loop
                        Err(e) => {
                            error!("Error receiving message from stream: {:?}", e);
                            return Err(InformerError::from(e));
                        }
                    }
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

        let merged_action = {
            let mut cache = self.action_cache.write().await;
            let cached_action = cache.get_mut(&action_name);
            if let Some(some_action) = cached_action {
                warn!("Submitting action {} and it's already in the cache!!! Existing {:?} <<<--->>> New: {:?}", action_name, some_action, action);
                some_action.merge_from_submit(&action);
                some_action.clone()
            } else {
                // don't need to write anything. return the original
                action
            }
        };
        warn!("Merged action: ===> {} {:?}", action_name, merged_action);

        // Store the completion event sender
        {
            let mut completion_events = self.completion_events.write().await;
            completion_events.insert(action_name.clone(), done_tx);
            warn!(
                "---------> Adding completion event in submit action {:?}",
                action_name
            );
        }

        // Add action to shared queue
        self.shared_queue.send(merged_action).await.map_err(|e| {
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
            warn!(
                "No completion event found for action---------------------: {}",
                action_name,
            );
            // Maybe the action hasn't started yet.
            return Ok(());
        }
        Ok(())
    }

    pub async fn stop(&self) {
        self.cancellation_token.cancel();
        if let Some(handle) = self.watch_handle.write().await.take() {
            warn!("Awaiting taken handle");
            let _ = handle.await;
            warn!("Taken handle finished...");
        } else {
            warn!("No handle to take ------------------------");
        }
        warn!("Stopped informer {:?}", self.parent_action_name);
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
        info!(">>> get_or_create_informer called for: {}", informer_name);
        let timeout = Duration::from_millis(100);

        // Check if exists (with read lock)
        {
            debug!("Acquiring read lock to check cache for: {}", informer_name);
            let map = self.cache.read().await;
            debug!("Read lock acquired, checking cache...");
            if let Some(informer) = map.get(&informer_name) {
                info!("CACHE HIT: Found existing informer for: {}", informer_name);
                let arc_informer = Arc::clone(informer);
                // Release read lock before waiting
                drop(map);
                debug!("Read lock released, waiting for ready...");
                Self::wait_for_ready(&arc_informer, timeout).await;
                info!("<<< Returning existing informer for: {}", informer_name);
                return arc_informer;
            }
            debug!("CACHE MISS: Informer not found in cache: {}", informer_name);
        }

        // Create new informer (with write lock)
        debug!(
            "Acquiring write lock to create informer for: {}",
            informer_name
        );
        let mut map = self.cache.write().await;
        info!("Write lock acquired for: {}", informer_name);

        // Double-check it wasn't created while we were waiting for write lock
        if let Some(informer) = map.get(&informer_name) {
            info!(
                "RACE: Informer was created while waiting for write lock: {}",
                informer_name
            );
            let arc_informer = Arc::clone(informer);
            drop(map);
            debug!("Write lock released after race condition");
            Self::wait_for_ready(&arc_informer, timeout).await;
            info!("<<< Returning race-created informer for: {}", informer_name);
            return arc_informer;
        }

        // Create and add to cache
        info!("CREATING new informer for: {}", informer_name);
        let informer = Arc::new(Informer::new(
            self.client.clone(),
            run_id.clone(),
            parent_action_name.to_string(),
            self.shared_queue.clone(),
        ));
        debug!("Informer object created, inserting into cache...");
        map.insert(informer_name.clone(), Arc::clone(&informer));
        info!("Informer inserted into cache: {}", informer_name);

        // Release write lock before starting (starting involves waiting)
        drop(map);
        debug!("Write lock released for: {}", informer_name);

        let me = Arc::clone(&informer);
        let failure_tx = self.failure_tx.clone();

        info!("Spawning watch task for: {}", informer_name);
        let _watch_handle = tokio::spawn(async move {
            const MAX_RETRIES: u32 = 10;
            const MIN_BACKOFF_SECS: f64 = 1.0;
            const MAX_BACKOFF_SECS: f64 = 30.0;

            let mut retries = 0;
            let mut last_error: Option<InformerError> = None;
            debug!("Watch task started for: {}", me.parent_action_name);

            while retries < MAX_RETRIES {
                if retries > 0 {
                    warn!(
                        "Informer watch retrying for {}, attempt {}/{}",
                        me.parent_action_name,
                        retries + 1,
                        MAX_RETRIES
                    );
                }

                let watch_result = me.watch_actions().await;
                match watch_result {
                    Ok(()) => {
                        // Clean exit (should only happen on cancellation)
                        info!("Watch completed cleanly for {}", me.parent_action_name);
                        last_error = None;
                        break;
                    }
                    Err(InformerError::Cancelled) => {
                        // Don't retry cancellations
                        info!(
                            "Watch cancelled for {}, exiting without retry",
                            me.parent_action_name
                        );
                        last_error = None;
                        break;
                    }
                    Err(err) if is_retryable_error(&err) => {
                        retries += 1;
                        last_error = Some(err.clone());

                        warn!(
                            "Watch failed for {} (retry {}/{}): {:?}",
                            me.parent_action_name, retries, MAX_RETRIES, err
                        );

                        if retries < MAX_RETRIES {
                            // Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s (capped)
                            let backoff = MIN_BACKOFF_SECS * 2_f64.powi((retries - 1) as i32);
                            let backoff = backoff.min(MAX_BACKOFF_SECS);
                            warn!("Backing off for {:.2}s before retry", backoff);
                            time::sleep(Duration::from_secs_f64(backoff)).await;
                        }
                    }
                    Err(err) => {
                        // Non-retryable error
                        error!(
                            "Non-retryable error for {}: {:?}",
                            me.parent_action_name, err
                        );
                        last_error = Some(err);
                        break;
                    }
                }
            }

            // Only send error if we have one (clean exits and cancellations set last_error = None)
            if let Some(err) = last_error {
                // We have an error - either exhausted retries or non-retryable
                error!(
                    "Informer watch failed for run {}, parent action {} (retries: {}/{}): {:?}",
                    me.run_id.name, me.parent_action_name, retries, MAX_RETRIES, err
                );

                let failure = InformerError::WatchFailed {
                    run_name: me.run_id.name.clone(),
                    parent_action_name: me.parent_action_name.clone(),
                    error_message: format!(
                        "Retries ({}/{}) exhausted. Last error: {}",
                        retries, MAX_RETRIES, err
                    ),
                };

                if let Err(e) = failure_tx.send(failure).await {
                    error!("Failed to send informer failure event: {:?}", e);
                }
            }
            // If last_error is None, it's a clean exit (Ok or Cancelled) - no error to send
        });

        // save the value and ignore the returned reference.
        debug!(
            "Acquiring write lock to save watch handle for: {}",
            informer_name
        );
        let _ = informer.watch_handle.write().await.insert(_watch_handle);
        info!("Watch handle saved for: {}", informer_name);

        // Optimistically wait for ready (sentinel) with timeout
        debug!("Waiting for informer to be ready: {}", informer_name);
        Self::wait_for_ready(&informer, timeout).await;

        info!(
            "<<< Returning newly created informer for: {}",
            informer_name
        );
        informer
    }

    pub async fn get(
        &self,
        run_id: &RunIdentifier,
        parent_action_name: &str,
    ) -> Option<Arc<Informer>> {
        let informer_name = InformerCache::mkname(&run_id.name, parent_action_name);
        debug!("InformerCache::get called for: {}", informer_name);
        let map = self.cache.read().await;
        let opt_informer = map.get(&informer_name).cloned();
        if opt_informer.is_some() {
            debug!("InformerCache::get - found: {}", informer_name);
        } else {
            debug!("InformerCache::get - not found: {}", informer_name);
        }
        opt_informer
    }

    /// Wait for informer to be ready with a timeout. If timeout occurs, set ready anyway
    /// and log a warning - this is optimistic, assuming the informer will become ready eventually.
    /// Once ready has been set, future calls return immediately without waiting.
    async fn wait_for_ready(informer: &Arc<Informer>, timeout: Duration) {
        debug!("wait_for_ready called for: {}", informer.parent_action_name);

        // Subscribe to notifications first, before checking ready
        // This ensures we don't miss a notification that happens between the check and the wait
        let ready_fut = informer.ready.notified();

        // Quick check - if already ready, return immediately
        if informer.is_ready.load(Ordering::Acquire) {
            info!(
                "Informer already ready for: {}",
                informer.parent_action_name
            );
            return;
        }

        debug!("Waiting for ready signal with timeout {:?}...", timeout);
        // Otherwise wait with timeout
        match tokio::time::timeout(timeout, ready_fut).await {
            Ok(_) => {
                info!(
                    "Informer ready signal received for: {}",
                    informer.parent_action_name
                );
            }
            Err(_) => {
                warn!(
                    "Informer ready TIMEOUT after {:?} for {}:{} - continuing optimistically",
                    timeout, informer.run_id.name, informer.parent_action_name
                );
                // Set ready anyway so future calls don't wait
                informer.is_ready.store(true, Ordering::Release);
            }
        }
    }

    pub async fn remove(
        &self,
        run_id: &RunIdentifier,
        parent_action_name: &str,
    ) -> Option<Arc<Informer>> {
        let informer_name = InformerCache::mkname(&run_id.name, parent_action_name);
        info!("InformerCache::remove called for: {}", informer_name);
        let mut map = self.cache.write().await;
        let opt_informer = map.remove(&informer_name);
        if opt_informer.is_some() {
            info!("InformerCache::remove - removed: {}", informer_name);
        } else {
            warn!("InformerCache::remove - not found: {}", informer_name);
        }
        opt_informer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            name: String::from("rchn685b8jgwtvz4k795"),
        };
        let (failure_tx, _failure_rx) = mpsc::channel::<InformerError>(1);

        let informer_cache = InformerCache::new(StateClient::Plain(client), tx.clone(), failure_tx);
        let informer = informer_cache.get_or_create_informer(&run_id, "a0").await;

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

    // cargo test --lib informer::tests:test_informer -- --nocapture --show-output
    #[test]
    fn test_informer() {
        init_tracing();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(informer_main());
    }
}
