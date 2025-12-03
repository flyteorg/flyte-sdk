use tonic::{Code, Status};
use tracing::debug;

/// Retry a streaming gRPC call once on authentication failure.
///
/// This is meant to be used when establishing a new stream. If the stream
/// fails with Unauthenticated, the auth middleware will have already refreshed
/// credentials, so we retry once with the fresh credentials.
///
/// # Example
/// ```rust
/// use flyte_controller_base::auth::retry_helper::retry_on_unauthenticated;
///
/// let stream = retry_on_unauthenticated(|| async {
///     client.bidirectional_stream(request.clone()).await
/// }).await?;
/// ```
pub async fn retry_on_unauthenticated<F, T, Fut>(f: F) -> Result<T, Status>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, Status>>,
{
    match f().await {
        Ok(result) => Ok(result),
        Err(status) if status.code() == Code::Unauthenticated => {
            debug!("Got Unauthenticated error, retrying once (middleware already refreshed credentials)");
            // Middleware already refreshed credentials, retry once
            f().await
        }
        Err(e) => Err(e),
    }
}

/// Retry wrapper that takes the number of retries for Unauthenticated errors
pub async fn retry_on_unauthenticated_n<F, T, Fut>(f: F, max_retries: usize) -> Result<T, Status>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, Status>>,
{
    let mut attempts = 0;
    loop {
        match f().await {
            Ok(result) => return Ok(result),
            Err(status) if status.code() == Code::Unauthenticated && attempts < max_retries => {
                attempts += 1;
                debug!("Got Unauthenticated error, retrying (attempt {}/{})", attempts, max_retries);
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}
