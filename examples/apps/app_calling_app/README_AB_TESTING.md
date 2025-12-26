# A/B Testing Example with Statsig

This example demonstrates how to implement A/B testing in Flyte apps using the Statsig SDK. A root app uses Statsig to determine which variant (App A or App B) to route requests to based on a user key.

## Architecture

The example consists of three FastAPI apps:

1. **App A** (`app_a`): First variant that processes requests with "fast-processing" algorithm
2. **App B** (`app_b`): Second variant that processes requests with "enhanced-processing" algorithm
3. **Root App** (`root_app`): Performs A/B testing and routes to either App A or App B

**Technical Note**: The statsig imports are done at runtime (inside functions) rather than at module level. This is because the `statsig-python-core` package is only available in the container image, not in the local development environment. This pattern allows the file to be imported and deployed without requiring statsig to be installed locally.

## How It Works

1. The Root App receives a request with a `message` and `user_key` parameter
2. It creates a Statsig user with the provided `user_key`
3. It checks the Statsig feature gate `variant_b` to determine which variant to use
4. Based on the gate result:
   - If enabled → calls App B
   - If disabled → calls App A
5. Returns the result along with A/B test metadata

## Setup

### 1. Get a Statsig API Key

1. Sign up at [statsig.com](https://www.statsig.com/)
2. Create a new project
3. Go to Settings → API Keys
4. Copy your Server Secret Key

### 2. Create a Feature Gate

1. In Statsig dashboard, go to Feature Gates
2. Create a new gate named `variant_b`
3. Configure targeting rules (e.g., 50% rollout, specific user IDs, etc.)
4. Save and enable the gate

### 3. Configure the Secret

Use Flyte's secrets management to set your Statsig API key:

```bash
# Set the secret in your Flyte deployment
flyte secrets set statsig-api-key STATSIG_API_KEY="your-secret-key-here"
```

For testing without Statsig, the example uses a default test key.

## Usage

### Deploy the App

```bash
cd examples/apps/app_calling_app
python ab_testing.py
```

### Test the A/B Testing Endpoint

```bash
# Test with different user keys to see A/B test routing
curl '<endpoint>/process/hello?user_key=user123'
curl '<endpoint>/process/world?user_key=user456'
curl '<endpoint>/process/test?user_key=user789'
```

Example response:

```json
{
  "ab_test_result": {
    "user_key": "user123",
    "selected_variant": "A",
    "gate_name": "variant_b"
  },
  "response": {
    "variant": "A",
    "message": "App A processed: hello",
    "algorithm": "fast-processing"
  }
}
```

### Get App Endpoints

```bash
curl '<endpoint>/endpoints'
```

This returns the direct endpoints for App A and App B if you want to test them individually:

```json
{
  "app_a_endpoint": "https://app-a-variant.example.com",
  "app_b_endpoint": "https://app-b-variant.example.com"
}
```

## Statsig Configuration

### Feature Gate Targeting

In the Statsig dashboard, you can configure targeting rules for the `variant_b` gate:

- **Percentage Rollout**: Gradually roll out App B to a percentage of users
- **User ID Targeting**: Route specific users to App B
- **Custom Rules**: Use user attributes for complex targeting

### Metrics and Analysis

Statsig automatically tracks:
- Exposure events (which users saw which variant)
- Gate check counts
- User distribution across variants

You can add custom metrics in Statsig to track business KPIs for each variant.

## Customization

### Change the A/B Test Logic

To use a different Statsig feature (experiment, dynamic config, etc.):

```python
# Instead of check_gate:
experiment = statsig.get_experiment(user, "my_experiment")
variant = experiment.get("variant", "A")

# Or use dynamic config:
config = statsig.get_config(user, "my_config")
use_variant_b = config.get("use_variant_b", False)
```

### Add More Variants

To support A/B/C testing:

1. Create `app_c` and `env_c`
2. Modify the logic to check multiple gates or use experiments
3. Add routing logic for the third variant

### Use Real Business Logic

Replace the simple `process_a` and `process_b` functions with your actual app logic:

```python
@app_a.post("/analyze")
async def analyze_a(data: RequestData) -> AnalysisResult:
    # Your original algorithm
    return old_algorithm(data)

@app_b.post("/analyze")
async def analyze_b(data: RequestData) -> AnalysisResult:
    # Your new improved algorithm
    return new_algorithm(data)
```

## Best Practices

1. **Consistent User Keys**: Use stable identifiers (user ID, session ID) for consistent bucketing
2. **Track Metrics**: Add Statsig events to track conversion, revenue, or other KPIs
3. **Gradual Rollout**: Start with small percentage and increase as you gain confidence
4. **Statistical Significance**: Run tests long enough to reach statistical significance
5. **Isolate Changes**: Test one change at a time for clear attribution

## Troubleshooting

### Gate Always Returns False

- Check that the gate is enabled in Statsig dashboard
- Verify your API key is correct in the Flyte secret
- Ensure the gate name matches exactly ("variant_b")

### Cannot Connect to Statsig

- Check your internet connection
- Verify the API key is set correctly
- Check Statsig service status

### Inconsistent Bucketing

- Ensure you're using the same user key consistently
- Check for typos in user_key parameter
- Verify Statsig configuration hasn't changed mid-test
