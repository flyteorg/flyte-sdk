use std::sync::Arc;
use std::task::{Context, Poll};
use tonic::body::BoxBody;
use tonic::transport::Channel;
use tower::{Layer, Service, ServiceExt};
use tracing::{error, warn};

use super::client_credentials::ClientCredentialsAuthenticator;

/// Tower layer that adds authentication to gRPC requests
#[derive(Clone)]
pub struct AuthLayer {
    authenticator: Arc<ClientCredentialsAuthenticator>,
    channel: Channel,
}

impl AuthLayer {
    pub fn new(authenticator: Arc<ClientCredentialsAuthenticator>, channel: Channel) -> Self {
        Self {
            authenticator,
            channel,
        }
    }
}

impl<S> Layer<S> for AuthLayer {
    type Service = AuthService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuthService {
            inner,
            authenticator: self.authenticator.clone(),
            channel: self.channel.clone(),
        }
    }
}

/// Tower service that intercepts requests to add authentication
#[derive(Clone)]
pub struct AuthService<S> {
    inner: S,
    authenticator: Arc<ClientCredentialsAuthenticator>,
    channel: Channel,
}

impl<S> Service<http::Request<BoxBody>> for AuthService<S>
where
    S: Service<http::Request<BoxBody>, Response = http::Response<BoxBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    S::Error: Into<Box<dyn std::error::Error + Send + Sync>> + Send,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: http::Request<BoxBody>) -> Self::Future {
        let authenticator = self.authenticator.clone();
        let channel = self.channel.clone();
        let mut inner = self.inner.clone();

        Box::pin(async move {
            // Get credentials and add auth header
            match authenticator.get_credentials(channel.clone()).await {
                Ok(creds) => {
                    let header_key = authenticator.get_header_key().await;
                    let token_value = format!("Bearer {}", creds.access_token);

                    warn!("Adding auth header: {}", header_key);

                    // Insert the authorization header
                    if let Ok(header_value) = token_value.parse::<http::HeaderValue>() {
                        request
                            .headers_mut()
                            .insert(http::header::AUTHORIZATION, header_value);
                    } else {
                        warn!("Failed to parse auth token as header value");
                    }
                }
                Err(e) => {
                    warn!("Failed to get credentials: {}", e);
                    // Continue without auth - let the server reject it
                }
            }

            if let Err(e) = inner.ready().await {
                error!("Inner service failed to become ready!!!");
                // Return the error from the inner service's ready check
                return Err(e);
            }

            // Make the request
            let result = inner.call(request).await;

            // Check for 401/Unauthenticated and refresh credentials for next time
            if let Ok(ref response) = result {
                if response.status() == http::StatusCode::UNAUTHORIZED {
                    warn!("Got 401, refreshing credentials for next request");

                    // Refresh credentials in background so next request will have fresh creds
                    if let Err(e) = authenticator.refresh_credentials(channel.clone()).await {
                        warn!("Failed to refresh credentials: {}", e);
                    }
                }
            }

            result
        })
    }
}
