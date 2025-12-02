// Generated protobuf files for Flyte IDL
// Only including the files needed for authentication and basic operations

#[path = "flyteidl.service.rs"]
pub mod service;

// Re-export the auth-related types and services for convenience
pub use service::auth_metadata_service_client::AuthMetadataServiceClient;
pub use service::OAuth2MetadataRequest;
pub use service::OAuth2MetadataResponse;
pub use service::PublicClientAuthConfigRequest;
pub use service::PublicClientAuthConfigResponse;
