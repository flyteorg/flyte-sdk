use thiserror::Error;

use crate::auth::AuthConfigError;

#[derive(Error, Debug)]
pub enum ControllerError {
    #[error("Bad context: {0}")]
    BadContext(String),
    #[error("Runtime error: {0}")]
    RuntimeError(String),
    #[error("System error: {0}")]
    SystemError(String),
    #[error("gRPC error: {0}")]
    GrpcError(#[from] tonic::Status),
    #[error("Task error: {0}")]
    TaskError(String),
    #[error("Informer error: {0}")]
    Informer(#[from] InformerError),
}

impl From<tonic::transport::Error> for ControllerError {
    fn from(err: tonic::transport::Error) -> Self {
        ControllerError::SystemError(format!("Transport error: {:?}", err))
    }
}

impl From<AuthConfigError> for ControllerError {
    fn from(err: AuthConfigError) -> Self {
        ControllerError::SystemError(err.to_string())
    }
}

#[derive(Error, Debug)]
pub enum InformerError {
    #[error("Informer watch failed for run {run_name}, parent action {parent_action_name}: {error_message}")]
    WatchFailed {
        run_name: String,
        parent_action_name: String,
        error_message: String,
    },
}
