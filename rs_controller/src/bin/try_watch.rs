/// Test binary to watch action updates from the Flyte API
///
/// Usage:
///   _UNION_EAGER_API_KEY=your_api_key cargo run --bin try_watch

use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tower::ServiceBuilder;
use tracing::{error, info, warn};

use flyte_controller_base::auth::{AuthConfig, AuthLayer, ClientCredentialsAuthenticator};
use flyte_controller_base::error::ControllerError;

use flyteidl2::flyteidl::common::{ActionIdentifier, RunIdentifier};
use flyteidl2::flyteidl::workflow::state_service_client::StateServiceClient;
use flyteidl2::flyteidl::workflow::watch_request::Filter;
use flyteidl2::flyteidl::workflow::WatchRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

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
    let endpoint = flyte_controller_base::core::create_tls_endpoint(static_endpoint, domain).await?;
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
                                return Ok(());
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
    Err(format!("Max watch retries ({}) exceeded", max_watch_retries).into())
}
