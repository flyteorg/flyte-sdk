import os
import typing
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

import flyte
from flyte.app.extras import FastAPIAppEnvironment


class StatsigClient:
    """Singleton to manage Statsig client lifecycle."""

    _instance: "StatsigClient | None" = None
    _statsig = None

    @classmethod
    def initialize(cls, api_key: str):
        """Initialize Statsig client (call during lifespan startup)."""
        if cls._instance is None:
            cls._instance = cls()

        # Import statsig at runtime (only available in container)
        from statsig_python_core import Statsig

        cls._statsig = Statsig(api_key)
        cls._statsig.initialize().wait()

    @classmethod
    def get_client(cls):
        """Get the initialized Statsig instance."""
        if cls._statsig is None:
            raise RuntimeError("StatsigClient not initialized. Call initialize() first.")
        return cls._statsig

    @classmethod
    def shutdown(cls):
        """Shutdown Statsig client (call during lifespan shutdown)."""
        if cls._statsig is not None:
            cls._statsig.shutdown()
            cls._statsig = None
            cls._instance = None


# Image with statsig-python-core for A/B testing
image = flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn", "httpx", "statsig-python-core")

# App A - First variant
app_a = FastAPI(
    title="App A",
    description="Variant A for A/B testing",
)

# App B - Second variant
app_b = FastAPI(
    title="App B",
    description="Variant B for A/B testing",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize and shutdown Statsig for A/B testing."""
    # Startup: Initialize Statsig using singleton
    api_key = os.getenv("STATSIG_API_KEY", None)
    if api_key is None:
        raise RuntimeError(f"StatsigClient API Key not set. ENV vars {os.environ}")
    StatsigClient.initialize(api_key)

    yield

    # Shutdown: Cleanup Statsig
    StatsigClient.shutdown()


# Root App - Performs A/B testing and routes to A or B
root_app = FastAPI(
    title="Root App - A/B Testing",
    description="Routes requests to App A or App B based on Statsig A/B test",
    lifespan=lifespan,
)

env_a = FastAPIAppEnvironment(
    name="app-a-variant",
    app=app_a,
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)

env_b = FastAPIAppEnvironment(
    name="app-b-variant",
    app=app_b,
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)

env_root = FastAPIAppEnvironment(
    name="root-ab-testing-app",
    app=root_app,
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    depends_on=[env_a, env_b],
    secrets=flyte.Secret("statsig-api-key", as_env_var="STATSIG_API_KEY"),
)


# App A endpoints
@app_a.get("/process/{message}")
async def process_a(message: str) -> dict[str, str]:
    return {
        "variant": "A",
        "message": f"App A processed: {message}",
        "algorithm": "fast-processing",
    }


# App B endpoints
@app_b.get("/process/{message}")
async def process_b(message: str) -> dict[str, str]:
    return {
        "variant": "B",
        "message": f"App B processed: {message}",
        "algorithm": "enhanced-processing",
    }


# Root app A/B testing endpoint
@root_app.get("/process/{message}")
async def process_with_ab_test(message: str, user_key: str) -> dict[str, typing.Any]:
    """
    Process a message using A/B testing to determine which app to call.

    Args:
        message: The message to process
        user_key: User identifier for A/B test bucketing (e.g., user_id, session_id)

    Returns:
        Response from either App A or App B, plus metadata about which variant was used
    """
    # Import StatsigUser at runtime (only available in container)
    from statsig_python_core import StatsigUser

    # Get statsig client from singleton
    statsig = StatsigClient.get_client()

    # Create Statsig user with the provided key
    user = StatsigUser(user_id=user_key)

    # Check the feature gate "variant_b" to determine which variant
    # If gate is enabled, use App B; otherwise use App A
    use_variant_b = statsig.check_gate(user, "variant_b")

    # Call the appropriate app based on A/B test result
    async with httpx.AsyncClient() as client:
        if use_variant_b:
            endpoint = f"{env_b.endpoint}/process/{message}"
            response = await client.get(endpoint)
            result = response.json()
        else:
            endpoint = f"{env_a.endpoint}/process/{message}"
            response = await client.get(endpoint)
            result = response.json()

    # Add A/B test metadata to response
    return {
        "ab_test_result": {
            "user_key": user_key,
            "selected_variant": "B" if use_variant_b else "A",
            "gate_name": "variant_b",
        },
        "response": result,
    }


@root_app.get("/endpoints")
async def get_endpoints() -> dict[str, str]:
    """Get the endpoints for App A and App B."""
    return {
        "app_a_endpoint": env_a.endpoint,
        "app_b_endpoint": env_b.endpoint,
    }


@root_app.get("/")
async def index():
    """Serve the A/B testing demo HTML page."""
    from fastapi.responses import HTMLResponse

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>A/B Testing Demo - Statsig</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell,
                sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }

            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                padding: 40px;
                max-width: 600px;
                width: 100%;
            }

            h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 28px;
            }

            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 14px;
            }

            .form-group {
                margin-bottom: 20px;
            }

            label {
                display: block;
                margin-bottom: 8px;
                color: #555;
                font-weight: 500;
                font-size: 14px;
            }

            input {
                width: 100%;
                padding: 12px 16px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                transition: border-color 0.3s;
            }

            input:focus {
                outline: none;
                border-color: #667eea;
            }

            button {
                width: 100%;
                padding: 14px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }

            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }

            button:active {
                transform: translateY(0);
            }

            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }

            .result {
                margin-top: 30px;
                padding: 20px;
                border-radius: 12px;
                display: none;
            }

            .result.show {
                display: block;
            }

            .result.variant-a {
                background: #e3f2fd;
                border: 2px solid #2196f3;
            }

            .result.variant-b {
                background: #f3e5f5;
                border: 2px solid #9c27b0;
            }

            .result-header {
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .variant-badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 700;
            }

            .variant-a .variant-badge {
                background: #2196f3;
                color: white;
            }

            .variant-b .variant-badge {
                background: #9c27b0;
                color: white;
            }

            .result-content {
                margin-top: 10px;
            }

            .result-item {
                margin-bottom: 10px;
                padding: 10px;
                background: rgba(255, 255, 255, 0.8);
                border-radius: 6px;
            }

            .result-label {
                font-weight: 600;
                color: #555;
                font-size: 13px;
            }

            .result-value {
                color: #333;
                margin-top: 4px;
            }

            .error {
                background: #ffebee;
                border: 2px solid #f44336;
                color: #c62828;
                padding: 16px;
                border-radius: 8px;
                margin-top: 20px;
                display: none;
            }

            .error.show {
                display: block;
            }

            .info {
                background: #fff3e0;
                border-left: 4px solid #ff9800;
                padding: 12px 16px;
                margin-top: 20px;
                border-radius: 4px;
                font-size: 13px;
                color: #e65100;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 A/B Testing Demo</h1>
            <p class="subtitle">Test Statsig-powered variant selection</p>

            <form id="abTestForm">
                <div class="form-group">
                    <label for="message">Message to Process</label>
                    <input
                        type="text"
                        id="message"
                        name="message"
                        placeholder="e.g., hello, world, test"
                        required
                        value="hello"
                    >
                </div>

                <div class="form-group">
                    <label for="userKey">User Key (for A/B bucketing)</label>
                    <input
                        type="text"
                        id="userKey"
                        name="userKey"
                        placeholder="e.g., user123, session456"
                        required
                        value="user123"
                    >
                </div>

                <button type="submit" id="submitBtn">Run A/B Test</button>
            </form>

            <div id="result" class="result"></div>
            <div id="error" class="error"></div>

            <div class="info">
                💡 <strong>Tip:</strong> Try different user keys to see how Statsig routes to different variants.
                The same user key will always get the same variant (consistent bucketing).
            </div>
        </div>

        <script>
            const form = document.getElementById('abTestForm');
            const resultDiv = document.getElementById('result');
            const errorDiv = document.getElementById('error');
            const submitBtn = document.getElementById('submitBtn');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();

                const message = document.getElementById('message').value;
                const userKey = document.getElementById('userKey').value;

                // Reset previous results
                resultDiv.classList.remove('show', 'variant-a', 'variant-b');
                errorDiv.classList.remove('show');
                submitBtn.disabled = true;
                submitBtn.textContent = 'Processing...';

                try {
                    const response =
                        await fetch(`/process/${encodeURIComponent(message)}?user_key=${encodeURIComponent(userKey)}`);

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const data = await response.json();

                    // Display result
                    const variant = data.ab_test_result.selected_variant;
                    const variantClass = `variant-${variant.toLowerCase()}`;

                    resultDiv.className = `result show ${variantClass}`;
                    resultDiv.innerHTML = `
                        <div class="result-header">
                            <span>A/B Test Result</span>
                            <span class="variant-badge">Variant ${variant}</span>
                        </div>
                        <div class="result-content">
                            <div class="result-item">
                                <div class="result-label">User Key</div>
                                <div class="result-value">${data.ab_test_result.user_key}</div>
                            </div>
                            <div class="result-item">
                                <div class="result-label">Selected Variant</div>
                                <div class="result-value">Variant ${variant}
                                    (Gate: ${data.ab_test_result.gate_name})</div>
                            </div>
                            <div class="result-item">
                                <div class="result-label">Response from App ${variant}</div>
                                <div class="result-value">${data.response.message}</div>
                            </div>
                            <div class="result-item">
                                <div class="result-label">Algorithm</div>
                                <div class="result-value">${data.response.algorithm}</div>
                            </div>
                        </div>
                    `;

                } catch (error) {
                    errorDiv.textContent = `Error: ${error.message}`;
                    errorDiv.classList.add('show');
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Run A/B Test';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env_root)
    print("Deployed A/B Testing Root App")
    print("\nUsage:")
    print("  Open your browser to '<endpoint>/' to access the interactive demo")
    print("  Or use curl: curl '<endpoint>/process/hello?user_key=user123'")
    print("\nNote: Set STATSIG_API_KEY secret to use real Statsig A/B testing.")
    print("      Create a feature gate named 'variant_b' in your Statsig dashboard.")
