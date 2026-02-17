/// Test binary to list tasks from the Flyte API
///
/// Usage:
///   _UNION_EAGER_API_KEY=your_api_key cargo run --bin try_list_tasks
use std::sync::Arc;


use flyte_controller_base::{
    auth::{AuthConfig, AuthLayer, ClientCredentialsAuthenticator},
};
use flyteidl2::flyteidl::{
    common::{ListRequest, ProjectIdentifier},
    task::{list_tasks_request, task_service_client::TaskServiceClient, ListTasksRequest},
};
use tonic::Code;
use tower::ServiceBuilder;
use tracing::warn;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let api_key = std::env::var("_UNION_EAGER_API_KEY").unwrap_or_else(|_| {
        warn!("_UNION_EAGER_API_KEY env var not set, using empty string");
        String::new()
    });

    let auth_config = AuthConfig::new_from_api_key(api_key.as_str())?;
    let endpoint = auth_config.endpoint.clone();
    let static_endpoint = endpoint.clone().leak();
    let channel = flyte_controller_base::core::create_tls_channel(static_endpoint).await?;

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
    let final_result: Result<bool, String> = loop {
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
                break Err(format!("gRPC error: {}", status));
            }
        }
    };
    warn!("Finished try_list_tasks with result {:?}", final_result);
    final_result?;
    Ok(())
}
