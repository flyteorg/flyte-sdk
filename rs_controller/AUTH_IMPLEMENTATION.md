# Rust Authentication Implementation for Flyte

## Summary

I've implemented client credentials OAuth2 authentication for the Rust gRPC clients, modeled after the Python implementation. The implementation includes:

1. **Auth Module Structure** (`src/auth/`)
   - `config.rs` - Auth configuration and helper traits
   - `token_client.rs` - OAuth2 token retrieval logic
   - `client_credentials.rs` - Client credentials authenticator with token caching
   - `interceptor.rs` - gRPC interceptor for adding auth headers and handling 401s

2. **Proto Module** (`src/proto/`)
   - Organized generated protobuf files from v1 Flyte IDL
   - Includes `AuthMetadataService` for fetching OAuth2 metadata

3. **Key Features**
   - Automatic token fetching on first request
   - Token caching with expiration tracking
   - Automatic refresh on 401/Unauthenticated errors
   - Thread-safe credential management using RwLock
   - Retry logic with automatic credential refresh

## Current Status

**Implemented but not fully compiling** - There are compilation issues with the generated proto files:

1. Some proto files have `#[derive(Copy)]` on structs with non-Copy fields (String)
2. There may be missing prost-types features needed for Timestamp handling
3. Some module visibility issues to resolve

## How It Works

### Authentication Flow

```
1. Client creates AuthConfig with endpoint, client_id, client_secret
2. ClientCredentialsAuthenticator is created
3. On first gRPC call:
   a. Authenticator fetches OAuth2 metadata from AuthMetadataService
   b. Calls token endpoint with client credentials
   c. Caches the access token with expiration time
4. AuthInterceptor adds "Bearer {token}" to request metadata
5. If request returns 401:
   a. Interceptor triggers credential refresh
   b. Retries the request with new token
```

### Usage Example

```rust
use flyte_controller_base::auth::{AuthConfig, AuthInterceptor, ClientCredentialsAuthenticator};

// Create auth config
let auth_config = AuthConfig {
    endpoint: "dns:///flyte.example.com:443".to_string(),
    client_id: "your_client_id".to_string(),
    client_secret: "your_secret".to_string(),
    scopes: None,
    audience: None,
};

// Create authenticator
let authenticator = Arc::new(ClientCredentialsAuthenticator::new(auth_config));

// Connect to endpoint
let channel = Endpoint::from_shared(endpoint)?
    .connect()
    .await?;

// Create auth interceptor
let auth_interceptor = AuthInterceptor::new(authenticator, channel.clone());

// Make authenticated calls using the with_auth! macro
let response = with_auth!(
    auth_interceptor,
    my_client,
    my_method,
    request
)?;
```

## Files Created/Modified

### New Files
- `src/auth/mod.rs` - Auth module exports
- `src/auth/config.rs` - Configuration types
- `src/auth/token_client.rs` - OAuth2 token client
- `src/auth/client_credentials.rs` - Authenticator implementation
- `src/auth/interceptor.rs` - gRPC interceptor with retry logic
- `src/proto/mod.rs` - Proto module organization
- `src/lib_auth.rs` - Re-exports for external use
- `examples/simple_auth_test.rs` - Test script
- `examples/auth_test.rs` - Example with actual API calls

### Modified Files
- `Cargo.toml` - Added dependencies (reqwest, serde, base64, urlencoding)
- `src/lib.rs` - Added auth and proto modules

## Next Steps to Fix Compilation

1. **Fix proto file issues:**
   - Remove `Copy` derives from structs with String fields in generated files
   - OR regenerate the proto files with correct options
   - OR use only the minimal auth-related protos

2. **Check prost-types dependency:**
   ```toml
   prost-types = { version = "0.12", features = ["std"] }
   ```
   May need to match the prost version exactly.

3. **Simplest fix:** Extract just the `AuthMetadataService` related types into a minimal hand-written proto module (I started this in `src/proto/auth_service.rs`)

## Testing

Once compilation is fixed, test with:

```bash
FLYTE_ENDPOINT=dns:///your-endpoint:443 \
FLYTE_CLIENT_ID=your_id \
FLYTE_CLIENT_SECRET=your_secret \
cargo run --example simple_auth_test
```

## References

- Python implementation: `/Users/ytong/go/src/github.com/flyteorg/flyte-sdk/src/flyte/remote/_client/auth/`
- Flytekit PR #2416: https://github.com/flyteorg/flytekit/pull/2416/files
- Proto definitions: https://github.com/flyteorg/flyte/tree/v2/flyteidl2
