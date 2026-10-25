use std::{
    path::PathBuf,
    time::{Duration, Instant, SystemTime},
};

use tokio::sync::RwLock;
use tracing::debug;

use super::{
    client_credentials::Credentials,
    errors::{AuthConfigError, TokenError},
};

/// Set on task pods by the platform when workload identity is enabled: the action attempt the pod
/// runs (`<org>/<project>/<domain>/<run>/<action>/<attempt>`) and its projected service-account token.
pub const WORKLOAD_ENV_VAR: &str = "_U_WORKLOAD";
pub const TOKEN_FILE_ENV_VAR: &str = "_U_WORKLOAD_TOKEN_FILE";
/// The header naming the attempt; the control plane accepts the token only for it.
pub const WORKLOAD_HEADER: &str = "x-union-workload";

/// How long a read of the token file is used before reading it again. The kubelet rotates the
/// token in place well before it expires (at 80% of its lifetime, an hour by default), so a minute
/// is always fresh.
const REREAD_AFTER: Duration = Duration::from_secs(60);

#[derive(Debug, Clone)]
struct ReadToken {
    credentials: Credentials,
    read_at: Instant,
}

/// Authenticates a task pod as the creator of its run, without any long-lived secret in the pod.
///
/// The pod presents its own projected Kubernetes service-account token and names the action
/// attempt it runs. The control plane accepts it only from the pod running that attempt, on the
/// cluster the attempt is leased to, and only for the run's project and domain.
#[derive(Debug)]
pub struct WorkloadTokenAuthenticator {
    workload: String,
    token_file: PathBuf,
    current: RwLock<Option<ReadToken>>,
}

impl WorkloadTokenAuthenticator {
    pub fn new(workload: impl Into<String>, token_file: impl Into<PathBuf>) -> Self {
        Self {
            workload: workload.into(),
            token_file: token_file.into(),
            current: RwLock::new(None),
        }
    }

    /// The authenticator the platform configured for this pod, from `_U_WORKLOAD` and
    /// `_U_WORKLOAD_TOKEN_FILE`.
    pub fn from_env() -> Result<Self, AuthConfigError> {
        let var = |name: &'static str| {
            std::env::var(name)
                .ok()
                .filter(|v| !v.is_empty())
                .ok_or(AuthConfigError::MissingEnvVar(name))
        };
        Ok(Self::new(var(WORKLOAD_ENV_VAR)?, var(TOKEN_FILE_ENV_VAR)?))
    }

    /// The attempt this pod speaks for, sent in `WORKLOAD_HEADER`.
    pub fn workload(&self) -> &str {
        &self.workload
    }

    async fn read(&self) -> Result<ReadToken, TokenError> {
        // Read again every time: the kubelet rotates the projected token in place.
        let raw = tokio::fs::read_to_string(&self.token_file)
            .await
            .map_err(|e| {
                TokenError::AuthError(format!(
                    "Cannot read the projected token at {}: {}",
                    self.token_file.display(),
                    e
                ))
            })?;
        let token = raw.trim();
        if token.is_empty() {
            return Err(TokenError::AuthError(format!(
                "The projected token at {} is empty",
                self.token_file.display()
            )));
        }
        debug!("Read the workload token for {}", self.workload);
        Ok(ReadToken {
            credentials: Credentials {
                access_token: token.to_string(),
                refresh_token: None,
                // Not the token's own expiry: when this read is next replaced.
                expires_at: SystemTime::now() + REREAD_AFTER,
            },
            read_at: Instant::now(),
        })
    }

    /// The current token, read again once the last read is `REREAD_AFTER` old.
    pub async fn get_credentials(&self) -> Result<Credentials, TokenError> {
        if let Some(t) = self.current.read().await.as_ref() {
            if t.read_at.elapsed() < REREAD_AFTER {
                return Ok(t.credentials.clone());
            }
        }
        let mut current = self.current.write().await;
        if let Some(t) = current.as_ref() {
            if t.read_at.elapsed() < REREAD_AFTER {
                return Ok(t.credentials.clone());
            }
        }
        let fresh = self.read().await?;
        let credentials = fresh.credentials.clone();
        *current = Some(fresh);
        Ok(credentials)
    }

    /// Read the token again now (after the control plane refused a request).
    pub async fn refresh_credentials(&self) -> Result<(), TokenError> {
        let fresh = self.read().await?;
        *self.current.write().await = Some(fresh);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal self-cleaning temp dir, to avoid a dev-dependency.
    struct Dir(PathBuf);

    impl Dir {
        fn new() -> Self {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static N: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "workload-token-test-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Dir(path)
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const WORKLOAD: &str = "acme/p/development/r1/a0/1";

    #[tokio::test]
    async fn reads_the_projected_token_and_caches_it() {
        let dir = Dir::new();
        let path = dir.0.join("token");
        std::fs::write(&path, "projected-sa-token-1\n").unwrap();
        let auth = WorkloadTokenAuthenticator::new(WORKLOAD, &path);

        assert_eq!(
            auth.get_credentials().await.unwrap().access_token,
            "projected-sa-token-1"
        );
        assert_eq!(auth.workload(), WORKLOAD);

        // Rotated in place: still the cached read until it is a minute old...
        std::fs::write(&path, "projected-sa-token-2").unwrap();
        assert_eq!(
            auth.get_credentials().await.unwrap().access_token,
            "projected-sa-token-1"
        );
        // ...or the control plane refused it.
        auth.refresh_credentials().await.unwrap();
        assert_eq!(
            auth.get_credentials().await.unwrap().access_token,
            "projected-sa-token-2"
        );
    }

    #[tokio::test]
    async fn rereads_a_stale_read() {
        let dir = Dir::new();
        let path = dir.0.join("token");
        std::fs::write(&path, "t1").unwrap();
        let auth = WorkloadTokenAuthenticator::new(WORKLOAD, &path);
        auth.get_credentials().await.unwrap();
        std::fs::write(&path, "t2").unwrap();
        if let Some(t) = auth.current.write().await.as_mut() {
            t.read_at = Instant::now() - REREAD_AFTER;
        }
        assert_eq!(auth.get_credentials().await.unwrap().access_token, "t2");
    }

    #[tokio::test]
    async fn refuses_a_missing_or_empty_token() {
        let dir = Dir::new();
        let path = dir.0.join("token");
        let auth = WorkloadTokenAuthenticator::new(WORKLOAD, &path);
        let err = auth.get_credentials().await.unwrap_err().to_string();
        assert!(err.contains("Cannot read the projected token"), "{err}");

        std::fs::write(&path, "  \n").unwrap();
        let err = auth.get_credentials().await.unwrap_err().to_string();
        assert!(err.contains("is empty"), "{err}");
    }
}
