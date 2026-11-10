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
    // The org is at one of its limits (e.g. max_actions_per_run): not retried, and surfaced to the parent as a
    // user error. The display prefix is matched by flyte/_internal/controllers/remote/_org_limits.py.
    #[error("Org limit exceeded: {0}")]
    OrgLimitExceeded(String),
}

/// The google.rpc.Status a gRPC error carries in `grpc-status-details-bin` (tonic's `Status::details`).
#[derive(Clone, PartialEq, prost::Message)]
pub(crate) struct RpcStatus {
    #[prost(int32, tag = "1")]
    pub code: i32,
    #[prost(string, tag = "2")]
    pub message: String,
    #[prost(message, repeated, tag = "3")]
    pub details: Vec<prost_types::Any>,
}

const ORG_LIMIT_EXCEEDED_TYPE: &str = "cloudidl.common.OrgLimitExceeded";

/// Returns the server's message when `status` is an org limit refusal: a `cloudidl.common.OrgLimitExceeded`
/// detail, recognized by its type URL since the SDK has no cloudidl protos.
pub fn org_limit_message(status: &tonic::Status) -> Option<String> {
    use prost::Message;
    if status.details().is_empty() {
        return None;
    }
    let rpc_status = RpcStatus::decode(status.details()).ok()?;
    rpc_status
        .details
        .iter()
        .any(|d| d.type_url.rsplit('/').next() == Some(ORG_LIMIT_EXCEEDED_TYPE))
        .then(|| status.message().to_string())
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

#[cfg(test)]
mod org_limit_tests {
    use super::*;
    use prost::Message;

    fn status_with_detail(type_url: &str) -> tonic::Status {
        let details = RpcStatus {
            code: tonic::Code::FailedPrecondition as i32,
            message: "over".to_string(),
            details: vec![prost_types::Any {
                type_url: type_url.to_string(),
                value: vec![],
            }],
        }
        .encode_to_vec();
        tonic::Status::with_details(
            tonic::Code::FailedPrecondition,
            "\"acme:p:d:r1\" would exceed org \"acme\"'s max_actions_per_run quota (3 current + 1 requested > 3)",
            details.into(),
        )
    }

    #[test]
    fn org_limit_detail_is_recognized() {
        let s = status_with_detail("type.googleapis.com/cloudidl.common.OrgLimitExceeded");
        let msg = org_limit_message(&s).expect("org limit refusal");
        assert!(msg.contains("max_actions_per_run"));
    }

    #[test]
    fn other_details_are_not_org_limits() {
        let s = status_with_detail("type.googleapis.com/google.rpc.ErrorInfo");
        assert!(org_limit_message(&s).is_none());
    }

    #[test]
    fn status_without_details_is_not_an_org_limit() {
        let s = tonic::Status::failed_precondition("wrong shard");
        assert!(org_limit_message(&s).is_none());
    }

    // Python's _org_limits.rust_client_error matches this prefix; keep them in step.
    #[test]
    fn org_limit_error_display_prefix() {
        let e = ControllerError::OrgLimitExceeded("over".to_string());
        assert_eq!(e.to_string(), "Org limit exceeded: over");
    }
}
