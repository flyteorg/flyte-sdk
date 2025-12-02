/// Configuration for authentication
#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub endpoint: String,
    pub client_id: String,
    pub client_secret: String,
    pub scopes: Option<Vec<String>>,
    pub audience: Option<String>,
}

/// Extension trait to add helper methods to the proto-generated PublicClientAuthConfigResponse
pub trait ClientConfigExt {
    fn header_key(&self) -> &str;
}

impl ClientConfigExt for crate::proto::PublicClientAuthConfigResponse {
    fn header_key(&self) -> &str {
        if self.authorization_metadata_key.is_empty() {
            "authorization"
        } else {
            &self.authorization_metadata_key
        }
    }
}
