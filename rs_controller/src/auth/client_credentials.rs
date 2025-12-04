use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tonic::transport::Channel;
use tracing::{debug, info};

use super::config::{AuthConfig, ClientConfigExt};
use super::token_client::{self, GrantType, TokenError, TokenResponse};
use crate::proto::{
    AuthMetadataServiceClient, OAuth2MetadataRequest, OAuth2MetadataResponse,
    PublicClientAuthConfigRequest, PublicClientAuthConfigResponse,
};

/// Stored credentials with expiration tracking
#[derive(Debug, Clone)]
pub struct Credentials {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: SystemTime,
}

impl Credentials {
    fn from_token_response(response: TokenResponse) -> Self {
        let expires_at = SystemTime::now() + Duration::from_secs(response.expires_in as u64);
        Self {
            access_token: response.access_token,
            refresh_token: response.refresh_token,
            expires_at,
        }
    }

    fn is_expired(&self) -> bool {
        // Consider expired if less than 60 seconds remaining
        SystemTime::now() + Duration::from_secs(60) >= self.expires_at
    }
}

/// Client credentials authenticator
pub struct ClientCredentialsAuthenticator {
    config: AuthConfig,
    credentials: Arc<RwLock<Option<Credentials>>>,
    client_config: Arc<RwLock<Option<PublicClientAuthConfigResponse>>>,
    oauth2_metadata: Arc<RwLock<Option<OAuth2MetadataResponse>>>,
}

impl ClientCredentialsAuthenticator {
    pub fn new(config: AuthConfig) -> Self {
        Self {
            config,
            credentials: Arc::new(RwLock::new(None)),
            client_config: Arc::new(RwLock::new(None)),
            oauth2_metadata: Arc::new(RwLock::new(None)),
        }
    }

    /// Get the client configuration from the auth metadata service
    async fn fetch_client_config(
        &self,
        channel: Channel,
    ) -> Result<PublicClientAuthConfigResponse, TokenError> {
        let mut client = AuthMetadataServiceClient::new(channel.clone());
        let request = tonic::Request::new(PublicClientAuthConfigRequest {});

        let response = client
            .get_public_client_config(request)
            .await
            .map_err(|e| TokenError::AuthError(format!("Failed to get client config: {}", e)))?;

        Ok(response.into_inner())
    }

    /// Get the OAuth2 metadata from the auth metadata service
    async fn fetch_oauth2_metadata(
        &self,
        channel: Channel,
    ) -> Result<OAuth2MetadataResponse, TokenError> {
        let mut client = AuthMetadataServiceClient::new(channel);
        let request = tonic::Request::new(OAuth2MetadataRequest {});

        let response = client
            .get_o_auth2_metadata(request)
            .await
            .map_err(|e| TokenError::AuthError(format!("Failed to get OAuth2 metadata: {}", e)))?;

        Ok(response.into_inner())
    }

    /// Refresh credentials using client credentials flow
    async fn refresh_credentials_internal(
        &self,
        channel: Channel,
    ) -> Result<Credentials, TokenError> {
        tracing::info!("🔄 refresh_credentials_internal: Starting...");
        // First, get the client config if we don't have it (cached)
        let client_config = {
            let config_lock = self.client_config.read().await;
            if let Some(cfg) = config_lock.as_ref() {
                tracing::info!("🔄 Using cached client_config");
                cfg.clone()
            } else {
                drop(config_lock);
                tracing::info!("🔄 Fetching client_config from auth service...");
                let cfg = self.fetch_client_config(channel.clone()).await?;
                tracing::info!("🔄 Got client_config response");
                let mut config_lock = self.client_config.write().await;
                *config_lock = Some(cfg.clone());
                cfg
            }
        };

        // Get OAuth2 metadata to find the token endpoint (cached)
        let oauth2_metadata = {
            let metadata_lock = self.oauth2_metadata.read().await;
            if let Some(metadata) = metadata_lock.as_ref() {
                metadata.clone()
            } else {
                drop(metadata_lock);
                let metadata = self.fetch_oauth2_metadata(channel).await?;
                let mut metadata_lock = self.oauth2_metadata.write().await;
                *metadata_lock = Some(metadata.clone());
                metadata
            }
        };

        debug!("Client credentials flow with client_id: {}", self.config.client_id);

        // Request the token
        let token_response = token_client::get_token(
            &oauth2_metadata.token_endpoint,
            &self.config.client_id,
            &self.config.client_secret,
            Some(client_config.scopes.as_slice()),
            Some(client_config.audience.as_str()),
            GrantType::ClientCredentials,
        )
        .await?;

        info!(
            "Retrieved new token, expires in {} seconds",
            token_response.expires_in
        );

        Ok(Credentials::from_token_response(token_response))
    }

    /// Get current credentials, refreshing if necessary
    pub async fn get_credentials(&self, channel: Channel) -> Result<Credentials, TokenError> {
        tracing::info!("🔐 get_credentials: Starting...");
        // Check if we have valid credentials
        {
            tracing::info!("🔐 get_credentials: Acquiring read lock...");
            let creds_lock = self.credentials.read().await;
            tracing::info!("🔐 get_credentials: Got read lock");
            if let Some(creds) = creds_lock.as_ref() {
                if !creds.is_expired() {
                    return Ok(creds.clone());
                }
            }
        }
        tracing::info!("🔐 get_credentials: Need to refresh, acquiring write lock...");

        // Need to refresh - acquire write lock
        let mut creds_lock = self.credentials.write().await;
        tracing::info!("🔐 get_credentials: Got write lock, calling refresh_credentials_internal...");

        // Double-check after acquiring write lock (another thread might have refreshed)
        if let Some(creds) = creds_lock.as_ref() {
            if !creds.is_expired() {
                return Ok(creds.clone());
            }
        }

        // Refresh the credentials
        let new_creds = self.refresh_credentials_internal(channel).await?;
        *creds_lock = Some(new_creds.clone());

        Ok(new_creds)
    }

    /// Force a refresh of credentials
    pub async fn refresh_credentials(&self, channel: Channel) -> Result<(), TokenError> {
        let new_creds = self.refresh_credentials_internal(channel).await?;
        let mut creds_lock = self.credentials.write().await;
        *creds_lock = Some(new_creds);
        Ok(())
    }

    /// Get the header key to use for authentication
    pub async fn get_header_key(&self) -> String {
        let config_lock = self.client_config.read().await;
        if let Some(cfg) = config_lock.as_ref() {
            // get rid of this
            cfg.header_key().to_string()
        } else {
            "authorization".to_string()
        }
    }
}
