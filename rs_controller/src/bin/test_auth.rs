/// Standalone authentication test binary
///
/// Usage:
///   FLYTE_ENDPOINT=dns:///your-endpoint:443 \
///   FLYTE_CLIENT_ID=your_id \
///   FLYTE_CLIENT_SECRET=your_secret \
///   cargo run --bin test_auth

use flyte_controller_base::auth::{AuthConfig, ClientCredentialsAuthenticator};
use std::env;
use std::sync::Arc;
use tonic::transport::Endpoint;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Flyte Client Credentials Authentication Test ===\n");

    let endpoint = env::var("FLYTE_ENDPOINT")
        .unwrap_or_else(|_| "dns:///localhost:8089".to_string());
    let client_id = env::var("FLYTE_CLIENT_ID")
        .expect("FLYTE_CLIENT_ID must be set");
    let client_secret = env::var("FLYTE_CLIENT_SECRET")
        .expect("FLYTE_CLIENT_SECRET must be set");

    println!("Endpoint: {}", endpoint);
    println!("Client ID: {}\n", client_id);

    let auth_config = AuthConfig {
        endpoint: endpoint.clone(),
        client_id,
        client_secret,
        scopes: None,
        audience: None,
    };

    let authenticator = Arc::new(ClientCredentialsAuthenticator::new(auth_config));

    println!("Connecting to endpoint...");
    let channel = Endpoint::from_shared(endpoint)?.connect().await?;
    println!("✓ Connected\n");

    println!("Fetching OAuth2 metadata and retrieving access token...");
    match authenticator.get_credentials(channel.clone()).await {
        Ok(creds) => {
            println!("✓ Successfully obtained access token!");
            let preview = &creds.access_token[..20.min(creds.access_token.len())];
            println!("  Token (first 20 chars): {}...", preview);
            println!("  Expires at: {:?}\n", creds.expires_at);

            println!("Testing cached credential retrieval...");
            match authenticator.get_credentials(channel).await {
                Ok(_) => println!("✓ Successfully retrieved cached credentials\n"),
                Err(e) => eprintln!("✗ Failed: {}\n", e),
            }

            println!("=== Test Complete ===");
            println!("Authentication is working correctly!");
            Ok(())
        }
        Err(e) => {
            eprintln!("✗ Failed to obtain access token: {}", e);
            Err(e.into())
        }
    }
}
