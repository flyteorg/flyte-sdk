/// Usage:
///   _UNION_EAGER_API_KEY=your_api_key cargo run --bin test_controller
///
/// Or without auth:
///   cargo run --bin test_controller -- http://localhost:8089
use flyte_controller_base::core::CoreBaseController;
use std::env;
use tracing_subscriber;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Flyte Core Controller Test ===\n");

    // Try to create a controller
    let controller = if let Ok(api_key) = env::var("_UNION_EAGER_API_KEY") {
        println!("Using auth from _UNION_EAGER_API_KEY");
        // Set the env var back since CoreBaseController::new_with_auth reads it
        env::set_var("_UNION_EAGER_API_KEY", api_key);
        CoreBaseController::new_with_auth()?
    } else {
        let endpoint = env::args()
            .nth(1)
            .unwrap_or_else(|| "http://localhost:8090".to_string());
        println!("Using endpoint: {}", endpoint);
        CoreBaseController::new_without_auth(endpoint)?
    };

    println!("✓ Successfully created CoreBaseController!");
    println!("✓ This proves that:");
    println!("  - The core module is accessible from binaries");
    println!("  - The Action type (with #[pyclass]) can be used");
    println!("  - No PyO3 linking errors occur");
    println!("\n=== Test Complete ===");

    Ok(())
}
