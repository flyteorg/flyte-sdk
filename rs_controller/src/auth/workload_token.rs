use std::{
    path::PathBuf,
    time::{Duration, Instant, SystemTime},
};

use serde::Deserialize;
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::{
    client_credentials::Credentials,
    errors::{AuthConfigError, TokenError},
};

/// Set on task pods by the platform when the dataplane STS is enabled.
pub const STS_ENDPOINT_ENV_VAR: &str = "_U_STS_ENDPOINT";
pub const STS_TOKEN_FILE_ENV_VAR: &str = "_U_STS_TOKEN_FILE";

const GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:token-exchange";
const JWT_TOKEN_TYPE: &str = "urn:ietf:params:oauth:token-type:jwt";
/// Refresh once this share of a token's life has passed, so a request never
/// starts with a token about to expire.
const REFRESH_AT: f64 = 0.8;
const STS_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Deserialize)]
struct ExchangeResponse {
    #[serde(default)]
    access_token: String,
    #[serde(default)]
    token_type: String,
    #[serde(default)]
    expires_in: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ExchangeError {
    #[serde(default)]
    error: String,
}

#[derive(Debug, Clone)]
struct CachedToken {
    credentials: Credentials,
    /// When to exchange again; `None` when the STS did not say how long the token lives.
    refresh_at: Option<Instant>,
}

impl CachedToken {
    fn is_fresh(&self, now: Instant) -> bool {
        self.refresh_at.is_none_or(|at| now < at)
    }
}

/// When a token obtained at `obtained_at` and living `expires_in` should be replaced.
fn refresh_deadline(obtained_at: Instant, expires_in: Option<u64>) -> Option<Instant> {
    expires_in.map(|secs| obtained_at + Duration::from_secs(secs).mul_f64(REFRESH_AT))
}

/// Authenticates a task pod as the creator of its run, without any long-lived secret in the pod.
///
/// The pod's projected service-account token (rotated by the kubelet, re-read on every refresh) is
/// exchanged at the dataplane STS (RFC 8693 token exchange) for a short-lived token the control
/// plane accepts for this run's project and domain only.
#[derive(Debug)]
pub struct WorkloadTokenAuthenticator {
    sts_endpoint: String,
    token_file: PathBuf,
    http: reqwest::Client,
    token: RwLock<Option<CachedToken>>,
}

impl WorkloadTokenAuthenticator {
    pub fn new(sts_endpoint: impl Into<String>, token_file: impl Into<PathBuf>) -> Self {
        // The STS is an in-cluster endpoint; it never goes through a proxy configured for the control plane.
        let http = reqwest::Client::builder()
            .timeout(STS_TIMEOUT)
            .no_proxy()
            .build()
            .unwrap_or_default();
        Self {
            sts_endpoint: sts_endpoint.into(),
            token_file: token_file.into(),
            http,
            token: RwLock::new(None),
        }
    }

    /// Build from `_U_STS_ENDPOINT` and `_U_STS_TOKEN_FILE`.
    pub fn from_env() -> Result<Self, AuthConfigError> {
        let read = |name: &'static str| {
            std::env::var(name)
                .ok()
                .filter(|v| !v.is_empty())
                .ok_or(AuthConfigError::MissingEnvVar(name))
        };
        let sts_endpoint = read(STS_ENDPOINT_ENV_VAR)?;
        let token_file = read(STS_TOKEN_FILE_ENV_VAR)?;
        Ok(Self::new(sts_endpoint, token_file))
    }

    async fn exchange(&self) -> Result<CachedToken, TokenError> {
        // Read on every refresh: the kubelet rotates the projected token in place.
        let subject_token = tokio::fs::read_to_string(&self.token_file)
            .await
            .map_err(|e| {
                TokenError::AuthError(format!(
                    "Cannot read the projected token at {}: {}",
                    self.token_file.display(),
                    e
                ))
            })?;
        let subject_token = subject_token.trim();
        if subject_token.is_empty() {
            return Err(TokenError::AuthError(format!(
                "The projected token at {} is empty",
                self.token_file.display()
            )));
        }

        let form = [
            ("grant_type", GRANT_TYPE),
            ("subject_token_type", JWT_TOKEN_TYPE),
            ("subject_token", subject_token),
        ];
        let response = self
            .http
            .post(&self.sts_endpoint)
            .header("Accept", "application/json")
            .form(&form)
            .send()
            .await
            .map_err(|e| {
                TokenError::AuthError(format!(
                    "The STS at {} is unreachable: {}",
                    self.sts_endpoint, e
                ))
            })?;

        let status = response.status();
        if !status.is_success() {
            let error = response
                .json::<ExchangeError>()
                .await
                .map(|e| e.error)
                .unwrap_or_default();
            return Err(TokenError::AuthError(
                format!(
                    "The STS refused the token exchange ({} {})",
                    status.as_u16(),
                    error
                )
                .trim_end()
                .to_string(),
            ));
        }

        let body: ExchangeResponse = response.json().await?;
        if body.access_token.is_empty() || !body.token_type.eq_ignore_ascii_case("bearer") {
            return Err(TokenError::AuthError(
                "The STS answered without a bearer token".to_string(),
            ));
        }

        let obtained_at = Instant::now();
        info!(
            "Obtained a workload token, expires in {:?}s",
            body.expires_in
        );
        let expires_at = match body.expires_in {
            Some(secs) => SystemTime::now() + Duration::from_secs(secs),
            None => SystemTime::UNIX_EPOCH + Duration::from_secs(u32::MAX as u64),
        };
        Ok(CachedToken {
            credentials: Credentials {
                access_token: body.access_token,
                refresh_token: None,
                expires_at,
            },
            refresh_at: refresh_deadline(obtained_at, body.expires_in),
        })
    }

    /// Get current credentials, exchanging again once 80% of the token's life has passed.
    pub async fn get_credentials(&self) -> Result<Credentials, TokenError> {
        {
            let token = self.token.read().await;
            if let Some(t) = token.as_ref().filter(|t| t.is_fresh(Instant::now())) {
                return Ok(t.credentials.clone());
            }
        }

        let mut token = self.token.write().await;
        // Double-check after acquiring the write lock (another task might have refreshed).
        if let Some(t) = token.as_ref().filter(|t| t.is_fresh(Instant::now())) {
            return Ok(t.credentials.clone());
        }
        debug!("Exchanging the projected token at {}", self.sts_endpoint);
        let fresh = self.exchange().await?;
        let credentials = fresh.credentials.clone();
        *token = Some(fresh);
        Ok(credentials)
    }

    /// Force a new exchange.
    pub async fn refresh_credentials(&self) -> Result<(), TokenError> {
        let fresh = self.exchange().await?;
        *self.token.write().await = Some(fresh);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    use super::*;

    /// A one-endpoint HTTP server: answers every request with the next canned
    /// `(status, body)` and records the request bodies it received.
    async fn fake_sts(responses: Vec<(u16, &'static str)>) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/oauth2/token", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_by_server = seen.clone();
        tokio::spawn(async move {
            for (status, body) in responses {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                // Read the head, then as much body as Content-Length says.
                let (head_end, content_length) = loop {
                    let n = sock.read(&mut chunk).await.unwrap();
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..pos]).to_lowercase();
                        let len = head
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .map(|v| v.trim().parse::<usize>().unwrap())
                            .unwrap_or(0);
                        break (pos + 4, len);
                    }
                };
                while buf.len() < head_end + content_length {
                    let n = sock.read(&mut chunk).await.unwrap();
                    buf.extend_from_slice(&chunk[..n]);
                }
                seen_by_server
                    .lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&buf[head_end..]).into_owned());
                let reply = format!(
                    "HTTP/1.1 {} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    body.len(),
                    body
                );
                sock.write_all(reply.as_bytes()).await.unwrap();
                sock.shutdown().await.ok();
            }
        });
        (url, seen)
    }

    fn token_file(content: &str) -> (tempdir::Dir, PathBuf) {
        let dir = tempdir::Dir::new();
        let path = dir.0.join("token");
        std::fs::write(&path, content).unwrap();
        (dir, path)
    }

    /// Minimal self-cleaning temp dir, to avoid a dev-dependency.
    mod tempdir {
        use std::path::PathBuf;

        pub struct Dir(pub PathBuf);

        impl Dir {
            pub fn new() -> Self {
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
    }

    const OK: &str = r#"{"access_token":"sts-token-1","token_type":"Bearer","expires_in":600}"#;
    const OK2: &str = r#"{"access_token":"sts-token-2","token_type":"bearer","expires_in":600}"#;

    fn form_field(body: &str, key: &str) -> Option<String> {
        url::form_urlencoded::parse(body.as_bytes())
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.into_owned())
    }

    #[tokio::test]
    async fn exchanges_the_projected_token() {
        let (url, seen) = fake_sts(vec![(200, OK)]).await;
        let (_dir, path) = token_file("projected-sa-token-1\n");
        let auth = WorkloadTokenAuthenticator::new(url, path);

        let creds = auth.get_credentials().await.unwrap();
        assert_eq!(creds.access_token, "sts-token-1");
        assert!(creds.refresh_token.is_none());

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(
            form_field(&seen[0], "grant_type").as_deref(),
            Some(GRANT_TYPE)
        );
        assert_eq!(
            form_field(&seen[0], "subject_token_type").as_deref(),
            Some(JWT_TOKEN_TYPE)
        );
        // Trailing newline from the projected file is stripped.
        assert_eq!(
            form_field(&seen[0], "subject_token").as_deref(),
            Some("projected-sa-token-1")
        );
    }

    #[tokio::test]
    async fn caches_a_fresh_token() {
        let (url, seen) = fake_sts(vec![(200, OK), (200, OK2)]).await;
        let (_dir, path) = token_file("t");
        let auth = WorkloadTokenAuthenticator::new(url, path);

        assert_eq!(
            auth.get_credentials().await.unwrap().access_token,
            "sts-token-1"
        );
        assert_eq!(
            auth.get_credentials().await.unwrap().access_token,
            "sts-token-1"
        );
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn refresh_rereads_the_rotated_token_file() {
        let (url, seen) = fake_sts(vec![(200, OK), (200, OK2)]).await;
        let (_dir, path) = token_file("projected-sa-token-1");
        let auth = WorkloadTokenAuthenticator::new(url, path.clone());

        auth.get_credentials().await.unwrap();
        std::fs::write(&path, "projected-sa-token-2").unwrap();
        auth.refresh_credentials().await.unwrap();

        assert_eq!(
            auth.get_credentials().await.unwrap().access_token,
            "sts-token-2"
        );
        let seen = seen.lock().unwrap();
        assert_eq!(
            form_field(&seen[1], "subject_token").as_deref(),
            Some("projected-sa-token-2")
        );
    }

    #[tokio::test]
    async fn exchanges_again_past_80_percent_of_the_lifetime() {
        let (url, seen) = fake_sts(vec![(200, OK), (200, OK2)]).await;
        let (_dir, path) = token_file("t");
        let auth = WorkloadTokenAuthenticator::new(url, path);

        auth.get_credentials().await.unwrap();
        // Age the cached token past its refresh deadline.
        if let Some(t) = auth.token.write().await.as_mut() {
            t.refresh_at = Some(Instant::now() - Duration::from_secs(1));
        }
        assert_eq!(
            auth.get_credentials().await.unwrap().access_token,
            "sts-token-2"
        );
        assert_eq!(seen.lock().unwrap().len(), 2);
    }

    #[test]
    fn refresh_deadline_is_80_percent_of_the_lifetime() {
        let now = Instant::now();
        assert_eq!(
            refresh_deadline(now, Some(600)),
            Some(now + Duration::from_secs(480))
        );
        assert_eq!(refresh_deadline(now, None), None);

        let without_expiry = CachedToken {
            credentials: Credentials {
                access_token: "t".to_string(),
                refresh_token: None,
                expires_at: SystemTime::now(),
            },
            refresh_at: None,
        };
        assert!(without_expiry.is_fresh(now + Duration::from_secs(86_400)));
    }

    #[tokio::test]
    async fn refusal_carries_the_oauth_error() {
        let (url, _) = fake_sts(vec![(403, r#"{"error":"access_denied"}"#)]).await;
        let (_dir, path) = token_file("t");
        let err = WorkloadTokenAuthenticator::new(url, path)
            .get_credentials()
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("refused the token exchange (403 access_denied)"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn rejects_a_non_bearer_answer() {
        let (url, _) = fake_sts(vec![(200, r#"{"access_token":"x","token_type":"mac"}"#)]).await;
        let (_dir, path) = token_file("t");
        let err = WorkloadTokenAuthenticator::new(url, path)
            .get_credentials()
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("without a bearer token"), "{err}");
    }

    #[tokio::test]
    async fn missing_or_empty_token_file_is_an_error() {
        let dir = tempdir::Dir::new();
        let auth = WorkloadTokenAuthenticator::new("http://127.0.0.1:1/", dir.0.join("absent"));
        let err = auth.get_credentials().await.unwrap_err().to_string();
        assert!(err.contains("Cannot read the projected token"), "{err}");

        let (_dir, path) = token_file("  \n");
        let auth = WorkloadTokenAuthenticator::new("http://127.0.0.1:1/", path);
        let err = auth.get_credentials().await.unwrap_err().to_string();
        assert!(err.contains("is empty"), "{err}");
    }

    #[test]
    fn from_env_needs_both_vars() {
        // No other test reads these variables, so setting them here does not race.
        std::env::remove_var(STS_ENDPOINT_ENV_VAR);
        std::env::remove_var(STS_TOKEN_FILE_ENV_VAR);
        assert!(matches!(
            WorkloadTokenAuthenticator::from_env(),
            Err(AuthConfigError::MissingEnvVar(STS_ENDPOINT_ENV_VAR))
        ));
        std::env::set_var(
            STS_ENDPOINT_ENV_VAR,
            "http://sts.union.svc:8095/oauth2/token",
        );
        assert!(matches!(
            WorkloadTokenAuthenticator::from_env(),
            Err(AuthConfigError::MissingEnvVar(STS_TOKEN_FILE_ENV_VAR))
        ));
        std::env::set_var(STS_TOKEN_FILE_ENV_VAR, "/var/run/secrets/union/sts/token");
        let auth = WorkloadTokenAuthenticator::from_env().unwrap();
        assert_eq!(auth.sts_endpoint, "http://sts.union.svc:8095/oauth2/token");
        assert_eq!(
            auth.token_file,
            PathBuf::from("/var/run/secrets/union/sts/token")
        );
        std::env::remove_var(STS_ENDPOINT_ENV_VAR);
        std::env::remove_var(STS_TOKEN_FILE_ENV_VAR);
    }

    #[tokio::test]
    async fn unreachable_sts_is_an_error() {
        // Bind then drop a listener so the port is closed.
        let addr = TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap()
            .local_addr()
            .unwrap();
        let (_dir, path) = token_file("t");
        let auth = WorkloadTokenAuthenticator::new(format!("http://{addr}/"), path);
        let err = auth.get_credentials().await.unwrap_err().to_string();
        assert!(err.contains("is unreachable"), "{err}");
    }
}
