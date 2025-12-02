use std::sync::Arc;
use tonic::transport::Channel;
use tonic::{Request, Status};
use tracing::{debug, warn};

use super::client_credentials::ClientCredentialsAuthenticator;

/// Auth interceptor that adds bearer token to requests and handles 401/Unauthenticated errors
#[derive(Clone)]
pub struct AuthInterceptor {
    authenticator: Arc<ClientCredentialsAuthenticator>,
    channel: Channel,
}

impl AuthInterceptor {
    pub fn new(authenticator: Arc<ClientCredentialsAuthenticator>, channel: Channel) -> Self {
        Self {
            authenticator,
            channel,
        }
    }

    /// Add authentication metadata to a request
    pub async fn add_auth_metadata<T>(&self, mut request: Request<T>) -> Result<Request<T>, Status> {
        match self.authenticator.get_credentials(self.channel.clone()).await {
            Ok(creds) => {
                let header_key = self.authenticator.get_header_key().await;
                let token_value = format!("Bearer {}", creds.access_token);

                debug!("Adding auth header: {}", header_key);
                let header_key_static: &'static str = Box::leak(header_key.into_boxed_str());
                request.metadata_mut().insert(
                    header_key_static,
                    token_value.parse().map_err(|e| {
                        Status::internal(format!("Failed to parse auth header: {}", e))
                    })?,
                );
                Ok(request)
            }
            Err(e) => {
                warn!("Failed to get credentials: {}", e);
                Err(Status::unauthenticated(format!("Failed to get credentials: {}", e)))
            }
        }
    }

    /// Handle authentication errors by refreshing credentials
    pub async fn handle_auth_error(&self) -> Result<(), Status> {
        debug!("Handling authentication error, refreshing credentials");
        self.authenticator
            .refresh_credentials(self.channel.clone())
            .await
            .map_err(|e| {
                Status::unauthenticated(format!("Failed to refresh credentials: {}", e))
            })
    }
}

/// Macro to create an authenticated client call with retry on 401
///
/// This macro handles the common pattern of:
/// 1. Adding auth metadata to the request
/// 2. Making the RPC call
/// 3. On UNAUTHENTICATED error, refreshing credentials and retrying once
#[macro_export]
macro_rules! with_auth {
    ($interceptor:expr, $client:expr, $method:ident, $request:expr) => {{
        use tonic::Code;

        // First attempt with current credentials
        let mut req = tonic::Request::new($request.clone());
        req = $interceptor.add_auth_metadata(req).await?;

        match $client.$method(req).await {
            Ok(response) => Ok(response),
            Err(e) if e.code() == Code::Unauthenticated || e.code() == Code::Unknown => {
                // Refresh credentials and retry
                $interceptor.handle_auth_error().await?;

                let mut req = tonic::Request::new($request);
                req = $interceptor.add_auth_metadata(req).await?;
                $client.$method(req).await
            }
            Err(e) => Err(e),
        }
    }};
}
