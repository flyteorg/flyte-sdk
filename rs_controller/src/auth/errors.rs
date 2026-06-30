use thiserror::Error;

#[derive(Error, Debug)]
pub enum AuthConfigError {
    #[error("Failed to decode base64: {0}")]
    Base64DecodeError(#[from] base64::DecodeError),

    #[error("Invalid API key format: expected 4 colon-separated parts, got {0}")]
    InvalidFormat(usize),

    #[error("Invalid UTF-8 in decoded API key")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),
}

#[derive(Error, Debug)]
pub enum TokenError {
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),
    #[error("Authentication error: {0}")]
    AuthError(String),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}
