use base64::{engine::general_purpose, Engine as _};
use reqwest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::auth::errors::TokenError;
use tracing::debug;


#[derive(Debug, Clone, Copy)]
pub enum GrantType {
    ClientCredentials,
}

impl GrantType {
    fn as_str(&self) -> &'static str {
        match self {
            GrantType::ClientCredentials => "client_credentials",
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TokenResponse {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    pub expires_in: i64,
    #[serde(default)]
    pub token_type: String,
}

/// Creates a basic authorization header from client ID and secret
pub fn get_basic_authorization_header(client_id: &str, client_secret: &str) -> String {
    let encoded_secret = urlencoding::encode(client_secret);
    let concatenated = format!("{}:{}", client_id, encoded_secret);
    let encoded = general_purpose::STANDARD.encode(concatenated.as_bytes());
    format!("Basic {}", encoded)
}

/// Retrieves an access token from the token endpoint
pub async fn get_token(
    token_endpoint: &str,
    client_id: &str,
    client_secret: &str,
    scopes: Option<&[String]>,
    audience: Option<&str>,
    grant_type: GrantType,
) -> Result<TokenResponse, TokenError> {
    let client = reqwest::Client::new();

    let authorization_header = get_basic_authorization_header(client_id, client_secret);

    let mut body = HashMap::new();
    body.insert("grant_type", grant_type.as_str().to_string());

    if let Some(scopes) = scopes {
        let scope_str = scopes.join(" ");
        body.insert("scope", scope_str);
    }

    if let Some(aud) = audience {
        body.insert("audience", aud.to_string());
    }

    debug!(
        "Requesting token from {} with grant_type {}",
        token_endpoint,
        grant_type.as_str()
    );

    let response = client
        .post(token_endpoint)
        .header("Authorization", authorization_header)
        .header("Cache-Control", "no-cache")
        .header("Accept", "application/json")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&body)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(TokenError::AuthError(format!(
            "Token request failed with status {}: {}",
            status, error_text
        )));
    }

    let token_response: TokenResponse = response.json().await?;
    debug!(
        "Retrieved new token, expires in {} seconds",
        token_response.expires_in
    );

    Ok(token_response)
}
