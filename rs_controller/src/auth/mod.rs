mod authenticator;
mod client_credentials;
mod config;
mod errors;
mod middleware;
mod token_client;
mod workload_token;

pub use authenticator::Authenticator;
pub use client_credentials::{ClientCredentialsAuthenticator, Credentials};
pub use config::AuthConfig;
pub use errors::{AuthConfigError, TokenError};
pub use middleware::{AuthLayer, AuthService};
pub use token_client::{get_token, GrantType};
pub use workload_token::{
    WorkloadTokenAuthenticator, STS_ENDPOINT_ENV_VAR, STS_TOKEN_FILE_ENV_VAR,
};
