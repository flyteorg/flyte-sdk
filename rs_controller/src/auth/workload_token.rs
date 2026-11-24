use std::{
    path::PathBuf,
    time::{Duration, Instant, SystemTime},
};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::Serialize;
use tokio::sync::RwLock;
use tracing::debug;

use super::{
    client_credentials::Credentials,
    errors::{AuthConfigError, TokenError},
};

/// Set on task pods by the platform when workload identity is enabled: the workload the pod runs
/// (an attempt `<org>/<project>/<domain>/<run>/<action>/<attempt>`, an app `app/...`, or a reusable
/// worker pod `pod/<org>/<project>/<domain>/<cluster>/<namespace>/<pod-name>`) and its projected
/// service-account token.
pub const WORKLOAD_ENV_VAR: &str = "_U_WORKLOAD";
pub const TOKEN_FILE_ENV_VAR: &str = "_U_WORKLOAD_TOKEN_FILE";

/// The prefix of a workload identity that names a pod (which may serve many actions) rather than a
/// single attempt or app.
const POD_WORKLOAD_PREFIX: &str = "pod/";
/// The credential's JWS header, exactly.
const CREDENTIAL_HEADER: &str = r#"{"typ":"union-workload","alg":"none"}"#;
/// The Actions service metadata that names the action a request is made for.
const RUN_HEADER: &str = "x-actions-run";
const PARENT_ACTION_HEADER: &str = "x-actions-parent-action";

#[derive(Serialize)]
struct CredentialAction<'a> {
    run: &'a str,
    name: &'a str,
}

#[derive(Serialize)]
struct CredentialPayload<'a> {
    token: &'a str,
    workload: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    action: Option<CredentialAction<'a>>,
}

/// The bearer credential for a workload token: an unsigned compact JWS,
/// `<header>.<payload>.` with base64url segments without padding. The header is
/// `{"typ":"union-workload","alg":"none"}`; the payload carries the projected token, the workload
/// and, when given, the action (`(run name, action name)`) the request is made for. The signature
/// is empty for now; the JWS shape leaves room to sign it later.
pub fn encode_workload_credential(
    token: &str,
    workload: &str,
    action: Option<(&str, &str)>,
) -> String {
    let payload = CredentialPayload {
        token,
        workload,
        action: action.map(|(run, name)| CredentialAction { run, name }),
    };
    // Serializing a struct of strings cannot fail.
    let payload = serde_json::to_vec(&payload).expect("workload credential payload serializes");
    format!(
        "{}.{}.",
        URL_SAFE_NO_PAD.encode(CREDENTIAL_HEADER),
        URL_SAFE_NO_PAD.encode(payload)
    )
}

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
/// The pod presents its own projected Kubernetes service-account token, together with the workload
/// it runs (and, for a reusable worker pod, the action a request is made for), as one bearer
/// credential. The control plane accepts it only from the pod running that workload, on the
/// cluster it is leased to, and only for the run's project and domain.
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

    /// The workload this pod speaks for.
    pub fn workload(&self) -> &str {
        &self.workload
    }

    /// The bearer credential for a request with these headers, wrapping `token` (the current read
    /// of the projected token). For a pod workload it names the action the request is made for,
    /// from the request's `x-actions-run` / `x-actions-parent-action` metadata, when both are set.
    pub fn credential(&self, token: &str, request_headers: &http::HeaderMap) -> String {
        let action = if self.workload.starts_with(POD_WORKLOAD_PREFIX) {
            let get = |name: &str| {
                request_headers
                    .get(name)
                    .and_then(|v| v.to_str().ok())
                    .filter(|v| !v.is_empty())
            };
            get(RUN_HEADER).zip(get(PARENT_ACTION_HEADER))
        } else {
            None
        };
        encode_workload_credential(token, &self.workload, action)
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
    const POD_WORKLOAD: &str = "pod/acme/p/development/c1/ns/worker-0";

    /// The decoded JWS header and payload of a credential, after checking its shape.
    fn decode(credential: &str) -> (String, serde_json::Value) {
        let parts: Vec<&str> = credential.split('.').collect();
        assert_eq!(parts.len(), 3, "{credential}");
        assert_eq!(parts[2], "", "unsigned");
        assert!(!credential.contains('='), "no padding");
        let header = String::from_utf8(URL_SAFE_NO_PAD.decode(parts[0]).unwrap()).unwrap();
        let payload = URL_SAFE_NO_PAD.decode(parts[1]).unwrap();
        (header, serde_json::from_slice(&payload).unwrap())
    }

    #[test]
    fn encodes_an_unsigned_compact_jws() {
        // Byte for byte what the Python SDK sends (tests/user_api/test_workload_token_auth.py).
        let attempt = encode_workload_credential("tok", WORKLOAD, None);
        assert_eq!(
            attempt,
            format!(
                "eyJ0eXAiOiJ1bmlvbi13b3JrbG9hZCIsImFsZyI6Im5vbmUifQ.{}.",
                URL_SAFE_NO_PAD
                    .encode(r#"{"token":"tok","workload":"acme/p/development/r1/a0/1"}"#)
            )
        );
        let (header, payload) = decode(&encode_workload_credential(
            "tok",
            POD_WORKLOAD,
            Some(("r1", "a0")),
        ));
        assert_eq!(header, r#"{"typ":"union-workload","alg":"none"}"#);
        assert_eq!(
            payload,
            serde_json::json!({
                "token": "tok",
                "workload": POD_WORKLOAD,
                "action": {"run": "r1", "name": "a0"},
            })
        );
    }

    #[test]
    fn names_the_action_only_for_a_pod_workload_with_actions_metadata() {
        let mut with_action = http::HeaderMap::new();
        with_action.insert(RUN_HEADER, "r1".parse().unwrap());
        with_action.insert(PARENT_ACTION_HEADER, "a0".parse().unwrap());
        let mut run_only = http::HeaderMap::new();
        run_only.insert(RUN_HEADER, "r1".parse().unwrap());

        let pod = WorkloadTokenAuthenticator::new(POD_WORKLOAD, "/unused");
        let (_, payload) = decode(&pod.credential("tok", &with_action));
        assert_eq!(
            payload["action"],
            serde_json::json!({"run": "r1", "name": "a0"})
        );
        let (_, payload) = decode(&pod.credential("tok", &run_only));
        assert!(payload.get("action").is_none(), "{payload}");
        let (_, payload) = decode(&pod.credential("tok", &http::HeaderMap::new()));
        assert!(payload.get("action").is_none(), "{payload}");

        for workload in [WORKLOAD, "app/acme/p/development/my-app"] {
            let auth = WorkloadTokenAuthenticator::new(workload, "/unused");
            let (_, payload) = decode(&auth.credential("tok", &with_action));
            assert_eq!(
                payload,
                serde_json::json!({"token": "tok", "workload": workload})
            );
        }
    }

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
