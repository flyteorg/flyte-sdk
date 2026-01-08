// Re-export auth module for use in examples and external crates
pub mod auth;
pub mod proto;

pub use auth::{AuthConfig, AuthInterceptor, ClientCredentialsAuthenticator, Credentials};
