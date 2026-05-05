use base64::{engine, Engine};

use crate::auth::errors::AuthConfigError;

/// Configuration for authentication
#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub endpoint: String,
    pub client_id: String,
    pub client_secret: String,
}

/// Extension trait to add helper methods to the proto-generated PublicClientAuthConfigResponse
pub trait ClientConfigExt {
    fn header_key(&self) -> &str;
}

// todo: get rid of this
impl ClientConfigExt for crate::proto::PublicClientAuthConfigResponse {
    fn header_key(&self) -> &str {
        if self.authorization_metadata_key.is_empty() {
            "authorization"
        } else {
            &self.authorization_metadata_key
        }
    }
}

impl AuthConfig {
    pub fn new_from_api_key(api_key: &str) -> Result<Self, AuthConfigError> {
        let decoded = engine::general_purpose::STANDARD.decode(api_key)?;
        let api_key_str = String::from_utf8(decoded)?;
        let split: Vec<_> = api_key_str.split(':').collect();

        if split.len() != 4 {
            return Err(AuthConfigError::InvalidFormat(split.len()));
        }

        let parts: [String; 4] = split
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let [endpoint, client_id, client_secret, _org] = parts;

        // the api key comes back just with the domain, we add https:// to it for rust rather than dns:///
        let endpoint = "https://".to_owned() + &endpoint;

        Ok(AuthConfig {
            endpoint,
            client_id,
            client_secret,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{engine, Engine};

    fn encode_api_key(parts: &[&str]) -> String {
        engine::general_purpose::STANDARD.encode(parts.join(":").as_bytes())
    }

    #[test]
    fn decodes_well_formed_api_key() {
        // Mirrors the format produced by the Python decode_api_key contract:
        // base64(endpoint:client_id:client_secret:org)
        let key = encode_api_key(&["example.com", "my-client", "s3cret", "my-org"]);

        let cfg = AuthConfig::new_from_api_key(&key).expect("should decode");

        assert_eq!(cfg.endpoint, "https://example.com");
        assert_eq!(cfg.client_id, "my-client");
        assert_eq!(cfg.client_secret, "s3cret");
    }

    #[test]
    fn endpoint_is_prefixed_with_https() {
        // The encoded payload contains the bare hostname; the decoder is
        // responsible for adding the scheme. Specifically NOT dns:/// — the
        // Rust tonic client wants a real URL.
        let key = encode_api_key(&["host.example", "id", "secret", "org"]);
        let cfg = AuthConfig::new_from_api_key(&key).unwrap();
        assert!(
            cfg.endpoint.starts_with("https://"),
            "expected https:// scheme, got {:?}",
            cfg.endpoint
        );
    }

    #[test]
    fn rejects_payload_with_wrong_number_of_parts() {
        // 3 parts — one short.
        let key = encode_api_key(&["host", "id", "secret"]);
        match AuthConfig::new_from_api_key(&key) {
            Err(AuthConfigError::InvalidFormat(n)) => assert_eq!(n, 3),
            other => panic!("expected InvalidFormat(3), got {:?}", other),
        }
    }

    #[test]
    fn rejects_invalid_base64() {
        let err =
            AuthConfig::new_from_api_key("not!!!base64@@@").expect_err("should reject bad base64");
        match err {
            AuthConfigError::Base64DecodeError(_) => {}
            other => panic!("expected Base64DecodeError, got {:?}", other),
        }
    }

    #[test]
    fn rejects_invalid_utf8() {
        // Valid base64 of an invalid UTF-8 sequence (lone continuation byte).
        let invalid_utf8: Vec<u8> = vec![0x80, 0x80, 0x80];
        let key = engine::general_purpose::STANDARD.encode(&invalid_utf8);
        match AuthConfig::new_from_api_key(&key) {
            Err(AuthConfigError::InvalidUtf8(_)) => {}
            other => panic!("expected InvalidUtf8, got {:?}", other),
        }
    }

    #[test]
    fn empty_org_is_accepted() {
        // The org field is intentionally ignored by the controller; an empty
        // string here is still a valid 4-part payload.
        let key = encode_api_key(&["host", "id", "secret", ""]);
        let cfg = AuthConfig::new_from_api_key(&key).expect("empty org should be fine");
        assert_eq!(cfg.endpoint, "https://host");
        assert_eq!(cfg.client_id, "id");
        assert_eq!(cfg.client_secret, "secret");
    }
}
