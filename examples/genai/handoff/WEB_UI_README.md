# Agent Handoff Web UI

A modern web interface for the agent handoff system, providing a Perplexity-like experience for querying and routing to specialized agents.

## Features

- 🤖 **Agent Browser**: View all 10 specialized agents with tags
- 💬 **Smart Query Input**: Enter custom queries or use example suggestions
- 🎯 **Real-time Results**: Watch as agents are evaluated and selected
- 📊 **Score Visualization**: See confidence scores for all candidate agents
- 🔗 **Flyte Integration**: Direct links to view executions in Flyte UI
- ⚡ **Async Execution**: Non-blocking workflow execution with `flyte.run.aio`

## Quick Start

### 1. Deploy the Web App

```bash
cd examples/genai/handoff
python app.py
```

This will:
- Build the Docker image with all dependencies
- Deploy the FastAPI app to Flyte
- Print the endpoint URL

### 2. Open in Browser

Navigate to the endpoint URL printed by the deployment:

```
🌐 Open in browser: https://your-app.example.com
```

## How It Works

### Architecture

```
┌─────────────┐
│  Web UI     │ (HTML/CSS/JS)
│  (Browser)  │
└──────┬──────┘
       │ HTTP/JSON
┌──────▼──────┐
│  FastAPI    │ app.py
│  Backend    │
└──────┬──────┘
       │ flyte.run.aio()
┌──────▼──────┐
│   Flyte     │ agent_handoff_workflow
│  Workflows  │
└─────────────┘
```

### Backend (app.py)

The FastAPI backend provides:

#### Endpoints

1. **`GET /`** - Serve HTML interface
2. **`GET /api/agents`** - List all available agents
3. **`GET /api/examples`** - Get example queries
4. **`POST /api/run`** - Trigger agent handoff workflow
5. **`GET /api/run/{run_id}/status`** - Get run status
6. **`GET /api/run/{run_id}/wait`** - Wait for run completion
7. **`GET /health`** - Health check

#### Key Features

**In-Cluster Initialization:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await flyte.init_in_cluster.aio()
    yield
```

**Async Run Execution:**
```python
run = await flyte.run.aio(
    agent_handoff_workflow,
    query=request.query,
    threshold=request.threshold
)
```

**Run Status Polling:**
```python
run_details = await remote.RunDetails.fetch(id=run_id)
status = run_details.status
if status == "SUCCEEDED":
    result = await run_details.result.aio()
```

### Frontend (index.html)

Modern single-page application with:

- **Agent Sidebar**: Browse all agents with tags
- **Example Chips**: Quick-select common queries
- **Query Textarea**: Enter custom queries
- **Threshold Slider**: Adjust confidence threshold (0.3 - 0.9)
- **Run Button**: Trigger handoff workflow
- **Results Display**:
  - Loading spinner during execution
  - Success: Selected agent + top scores
  - Failure: Error message + closest matches
  - Link to Flyte UI for detailed view

## User Interface

### Main Screen

```
┌─────────────────────────────────────────────────────────────┐
│                    🤖 Agent Handoff                         │
│        Intelligent routing to specialized agents            │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐  ┌─────────────────────────────────────────┐
│ Agents (10)  │  │ Try these example queries:              │
│              │  │ [Analyze sales data] [Train ML model]   │
│ Data Analyst │  │ [Setup CI/CD] [Security audit]          │
│ Code Reviewer│  │                                         │
│ ML Engineer  │  │ ┌─────────────────────────────────────┐ │
│ DevOps Agent │  │ │ Enter your query here...            │ │
│ ...          │  │ │                                     │ │
│              │  │ └─────────────────────────────────────┘ │
│              │  │                                         │
│              │  │ Threshold: 0.6 [===|====] [Run]        │
└──────────────┘  └─────────────────────────────────────────┘
```

### Results Display

**Successful Handoff:**
```
✓ Handoff Successful

🎯 Selected Agent: Data Analytics Agent
   Specializes in data analysis, SQL queries...
   [data] [analytics] [sql] [visualization]

📊 Analysis Details
   Extracted Tags: data, analytics, visualization
   Agents Evaluated: 3

🏆 Top Matching Agents
   Data Analytics Agent  [████████████] 85.3%
   Data Pipeline Agent   [████████    ] 67.2%
   ML Engineer Agent     [██████      ] 54.1%

View full execution in Flyte UI →
```

**No Match:**
```
⚠ No Match Found

No agent met the threshold (0.8)

📊 Closest Matches
   Data Analytics Agent  [████████    ] 74.2%
   Code Review Agent     [██████      ] 65.8%
   ...

💡 Try lowering the threshold or rephrasing your query

View execution in Flyte UI →
```

## Configuration

### Threshold Settings

- **0.3 - 0.5**: Very permissive (broad matches)
- **0.6**: Recommended default (balanced)
- **0.7 - 0.9**: Strict (only close matches)

### Timeout

Default timeout for run completion: 60 seconds

Adjust in the wait endpoint:
```python
@app.get("/api/run/{run_id}/wait")
async def wait_for_run(run_id: str, timeout: int = 60):
    ...
```

## API Usage

### Trigger Run

```bash
curl -X POST "https://your-app.example.com/api/run" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I need help analyzing sales data",
    "threshold": 0.6
  }'
```

Response:
```json
{
  "run_id": "f12345...",
  "url": "https://flyte.example.com/console/...",
  "status": "RUNNING"
}
```

### Check Status

```bash
curl "https://your-app.example.com/api/run/f12345.../status"
```

Response:
```json
{
  "run_id": "f12345...",
  "url": "https://flyte.example.com/console/...",
  "status": "SUCCEEDED",
  "result": {
    "handoff_successful": true,
    "selected_agent": {
      "id": "data-analyst",
      "name": "Data Analytics Agent",
      ...
    },
    "extracted_tags": ["data", "analytics"],
    "top_scores": [...]
  }
}
```

## Development

### Local Testing

```bash
# Install dependencies
uv sync

# Run locally (without Flyte)
uvicorn app:app --reload

# Access at http://localhost:8000
```

### Adding Custom Agents

Edit `agent_handoff.py` and add to `AGENT_REGISTRY`:

```python
Agent(
    id="your-agent",
    name="Your Custom Agent",
    description="Detailed description for semantic matching",
    tags=["tag1", "tag2", "tag3"]
)
```

The web UI will automatically load the new agent.

### Customizing UI

Edit `static/index.html`:
- Modify CSS in `<style>` section
- Update JavaScript in `<script>` section
- No build step required - changes are immediate

## Deployment

### Production Deployment

```bash
# Deploy to production domain
flyte deploy app.py --domain production

# Scale the deployment
flyte app update agent-handoff-ui --replicas 3
```

### Environment Variables

Set in the FastAPI environment:

```python
env = FastAPIAppEnvironment(
    name="agent-handoff-ui",
    env_vars={
        "LOG_LEVEL": "INFO",
        "TIMEOUT": "120",
    }
)
```

## Monitoring

### Health Check

```bash
curl https://your-app.example.com/health
```

Response:
```json
{"status": "healthy"}
```

### Logs

View logs in Flyte UI or via CLI:

```bash
flyte logs get agent-handoff-ui
```

## Troubleshooting

### "Failed to initialize Flyte"

Ensure the app is deployed in-cluster with proper credentials:
```python
await flyte.init_in_cluster.aio()
```

### "Run not found"

The run ID may be invalid or expired. Check the URL format.

### "Timeout waiting for run"

Increase the timeout parameter:
```javascript
const finalStatus = await waitForRun(runData.run_id, 120); // 2 minutes
```

### UI not loading

Check that `static/index.html` exists and is being served:
```bash
curl https://your-app.example.com/
```

## Performance

- **Page Load**: <1s (single HTML file, no external dependencies)
- **Query Execution**: ~900ms for 10 agents
- **Status Polling**: 2s intervals
- **Total UX**: ~3-5s from query to results

## Security

### Authentication

To add authentication, modify the FastAPI app:

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/api/run")
async def run_handoff(
    request: QueryRequest,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
):
    # Verify token
    ...
```

### CORS

If accessing from different domain:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],
    allow_methods=["*"],
)
```

## Future Enhancements

- [ ] WebSocket support for real-time status updates
- [ ] Agent performance analytics dashboard
- [ ] Query history and favorites
- [ ] Multi-agent handoff chains
- [ ] User feedback loop for improving routing

## License

See main Flyte SDK license.
