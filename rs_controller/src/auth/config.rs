use crate::auth::errors::AuthConfigError;
use base64::{engine, Engine};

/// Configuration for authentication
#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub endpoint: String,
    pub client_id: String,
    pub client_secret: String,
}

/// Extension trait to add helper methods to the proto-generated PublicClientAuthConfigResponse
pub trait ClientConfigExt {
    fn header_key(&self) -> &str;
}

// todo: get rid of this
impl ClientConfigExt for crate::proto::PublicClientAuthConfigResponse {
    fn header_key(&self) -> &str {
        if self.authorization_metadata_key.is_empty() {
            "authorization"
        } else {
            &self.authorization_metadata_key
        }
    }
}

impl AuthConfig {
    pub fn new_from_api_key(api_key: &str) -> Result<Self, AuthConfigError> {
        let decoded = engine::general_purpose::STANDARD.decode(api_key)?;
        let api_key_str = String::from_utf8(decoded)?;
        let split: Vec<_> = api_key_str.split(':').collect();

        if split.len() != 4 {
            return Err(AuthConfigError::InvalidFormat(split.len()));
        }

        let parts: [String; 4] = split
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let [endpoint, client_id, client_secret, _org] = parts;
        Ok(AuthConfig {
            endpoint,
            client_id,
            client_secret,
        })
    }
}
