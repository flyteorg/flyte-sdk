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
    GrpcError(#[from] Box<tonic::Status>),
    #[error("Task error: {0}")]
    TaskError(String),
    #[error("Informer error: {0}")]
    Informer(#[from] InformerError),
    // Error type that triggers retry with backoff
    #[error("Slow down error: {0}")]
    SlowDownError(String),
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

#[derive(Error, Debug, Clone)]
pub enum InformerError {
    #[error("Informer watch failed for run {run_name}, parent action {parent_action_name}: {error_message}")]
    WatchFailed {
        run_name: String,
        parent_action_name: String,
        error_message: String,
    },
    #[error("gRPC error in watch stream: {0}")]
    GrpcError(String),
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Failed to send action to queue: {0}")]
    QueueSendError(String),
    #[error("Watch cancelled")]
    Cancelled,
    #[error("Bad context: {0}")]
    BadContext(String),
}

impl From<tonic::Status> for InformerError {
    fn from(status: tonic::Status) -> Self {
        InformerError::GrpcError(format!("{:?}", status))
    }
}
