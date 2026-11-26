use tonic::transport::Channel;

use super::{
    client_credentials::{ClientCredentialsAuthenticator, Credentials},
    errors::TokenError,
    workload_token::WorkloadTokenAuthenticator,
};

/// The ways the controller can authenticate its gRPC calls.
#[derive(Debug)]
pub enum Authenticator {
    /// OAuth2 client credentials, decoded from the injected api key.
    ClientCredentials(ClientCredentialsAuthenticator),
    /// The pod's own projected service-account token (workload identity).
    WorkloadToken(WorkloadTokenAuthenticator),
}

impl Authenticator {
    /// Get current credentials, refreshing if necessary
    pub async fn get_credentials(&self, channel: Channel) -> Result<Credentials, TokenError> {
        match self {
            Authenticator::ClientCredentials(a) => a.get_credentials(channel).await,
            Authenticator::WorkloadToken(a) => a.get_credentials().await,
        }
    }

    /// Force a refresh of credentials
    pub async fn refresh_credentials(&self, channel: Channel) -> Result<(), TokenError> {
        match self {
            Authenticator::ClientCredentials(a) => a.refresh_credentials(channel).await,
            Authenticator::WorkloadToken(a) => a.refresh_credentials().await,
        }
    }

    /// Get the header key to use for authentication
    pub async fn get_header_key(&self) -> String {
        match self {
            Authenticator::ClientCredentials(a) => a.get_header_key().await,
            Authenticator::WorkloadToken(_) => "authorization".to_string(),
        }
    }
}

impl Authenticator {
    /// The bearer value for a request with these headers: the access token as is, or, for a
    /// workload token, the composite credential wrapping it (see `encode_workload_credential`).
    pub fn bearer_token(&self, creds: &Credentials, request_headers: &http::HeaderMap) -> String {
        match self {
            Authenticator::ClientCredentials(_) => creds.access_token.clone(),
            Authenticator::WorkloadToken(a) => a.credential(&creds.access_token, request_headers),
        }
    }
}

impl From<ClientCredentialsAuthenticator> for Authenticator {
    fn from(a: ClientCredentialsAuthenticator) -> Self {
        Authenticator::ClientCredentials(a)
    }
}

impl From<WorkloadTokenAuthenticator> for Authenticator {
    fn from(a: WorkloadTokenAuthenticator) -> Self {
        Authenticator::WorkloadToken(a)
    }
}
