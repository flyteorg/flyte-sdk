import json
import random

import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="network_graph",
)


@env.task(report=True)
async def generate_network_graph():
    # Generate sample network data
    network_data = generate_network_data()

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Simple Network Graph</title>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                font-family: 'Arial', sans-serif;
                background: linear-gradient(135deg, #1e1e2e 0%, #2d1b69 50%, #11998e 100%);
                color: white;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1 {{
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 30px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }}
            .network-container {{
                background: rgba(0, 0, 0, 0.3);
                border-radius: 15px;
                padding: 20px;
                margin: 20px 0;
                position: relative;
                height: 600px;
                overflow: hidden;
            }}
            .node {{
                position: absolute;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 10px;
                font-weight: bold;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                border: 2px solid rgba(255, 255, 255, 0.3);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            }}
            .node:hover {{
                transform: scale(1.1);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
            }}
            .node.server {{ background: #ff6b6b; }}
            .node.database {{ background: #4ecdc4; }}
            .node.client {{ background: #45b7d1; }}
            .node.api {{ background: #f9ca24; }}
            .node.service {{ background: #6c5ce7; }}
            .connection {{
                position: absolute;
                height: 2px;
                background: linear-gradient(90deg, rgba(56, 239, 125, 0.6), rgba(17, 153, 142, 0.6));
                transform-origin: left center;
                opacity: 0.7;
                border-radius: 1px;
            }}
            .legend {{
                display: flex;
                justify-content: center;
                gap: 30px;
                margin: 20px 0;
                flex-wrap: wrap;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 8px;
                background: rgba(0, 0, 0, 0.3);
                padding: 8px 15px;
                border-radius: 20px;
            }}
            .legend-color {{
                width: 20px;
                height: 20px;
                border-radius: 50%;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-card {{
                background: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                padding: 20px;
                text-align: center;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #38ef7d;
            }}
            @media (max-width: 768px) {{
                .node {{
                    width: 40px;
                    height: 40px;
                    font-size: 8px;
                }}
                .legend {{
                    flex-direction: column;
                    align-items: center;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🕸️ Network Graph Visualization</h1>

            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff6b6b;"></div>
                    <span>Server</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4ecdc4;"></div>
                    <span>Database</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #45b7d1;"></div>
                    <span>Client</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #f9ca24;"></div>
                    <span>API</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #6c5ce7;"></div>
                    <span>Service</span>
                </div>
            </div>

            <div class="network-container" id="networkContainer">
                <!-- Network nodes and connections will be generated here -->
            </div>

            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">15</div>
                    <div>Total Nodes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">28</div>
                    <div>Connections</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">5</div>
                    <div>Node Types</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">100%</div>
                    <div>Network Health</div>
                </div>
            </div>
        </div>

        <script>
            const networkData = {json.dumps(network_data)};

            function createNetwork() {{
                const container = document.getElementById('networkContainer');
                const containerRect = container.getBoundingClientRect();
                const width = containerRect.width - 40;
                const height = containerRect.height - 40;

                // Clear existing content
                container.innerHTML = '';

                // Create nodes
                networkData.nodes.forEach((node, index) => {{
                    const nodeEl = document.createElement('div');
                    nodeEl.className = `node ${{node.type}}`;
                    nodeEl.textContent = node.name;
                    nodeEl.title = `${{node.name}} (${{node.type}})`;

                    // Position nodes in a circle layout
                    const angle = (index / networkData.nodes.length) * 2 * Math.PI;
                    const radius = Math.min(width, height) * 0.35;
                    const centerX = width / 2;
                    const centerY = height / 2;

                    const x = centerX + radius * Math.cos(angle) - 30;
                    const y = centerY + radius * Math.sin(angle) - 30;

                    nodeEl.style.left = x + 'px';
                    nodeEl.style.top = y + 'px';

                    container.appendChild(nodeEl);
                }});

                // Create connections
                networkData.links.forEach(link => {{
                    const sourceNode = networkData.nodes.find(n => n.id === link.source);
                    const targetNode = networkData.nodes.find(n => n.id === link.target);

                    if (sourceNode && targetNode) {{
                        const sourceIndex = networkData.nodes.indexOf(sourceNode);
                        const targetIndex = networkData.nodes.indexOf(targetNode);

                        const sourceAngle = (sourceIndex / networkData.nodes.length) * 2 * Math.PI;
                        const targetAngle = (targetIndex / networkData.nodes.length) * 2 * Math.PI;
                        const radius = Math.min(width, height) * 0.35;
                        const centerX = width / 2;
                        const centerY = height / 2;

                        const sourceX = centerX + radius * Math.cos(sourceAngle);
                        const sourceY = centerY + radius * Math.sin(sourceAngle);
                        const targetX = centerX + radius * Math.cos(targetAngle);
                        const targetY = centerY + radius * Math.sin(targetAngle);

                        const connection = document.createElement('div');
                        connection.className = 'connection';

                        const length = Math.sqrt(Math.pow(targetX - sourceX, 2) + Math.pow(targetY - sourceY, 2));
                        const angle = Math.atan2(targetY - sourceY, targetX - sourceX) * 180 / Math.PI;

                        connection.style.left = sourceX + 'px';
                        connection.style.top = sourceY + 'px';
                        connection.style.width = length + 'px';
                        connection.style.transform = `rotate(${{angle}}deg)`;
                        connection.style.opacity = Math.min(link.weight / 10, 0.8);

                        container.appendChild(connection);
                    }}
                }});
            }}

            // Initialize on load and resize
            document.addEventListener('DOMContentLoaded', createNetwork);
            window.addEventListener('resize', createNetwork);
        </script>
    </body>
    </html>
    """

    await flyte.report.replace.aio(html_content)
    await flyte.report.flush.aio()


def generate_network_data():
    """Generate sample network data with nodes and links"""
    node_types = ["server", "database", "client", "api", "service"]

    # Generate nodes
    nodes = []
    node_names = [
        "Web Server",
        "Database Primary",
        "Load Balancer",
        "API Gateway",
        "Auth Service",
        "User Service",
        "Payment API",
        "Cache Server",
        "File Storage",
        "Analytics DB",
        "Mobile Client",
        "Web Client",
        "Admin Panel",
        "Monitoring",
        "Log Server",
        "Message Queue",
        "Search Engine",
        "CDN",
        "Backup Server",
        "Test Environment",
    ]

    for i, name in enumerate(node_names):
        node = {
            "id": f"node_{i}",
            "name": name,
            "type": random.choice(node_types),
            "size": random.randint(8, 25),
            "weight": random.randint(1, 10),
            "status": random.choice(["Active", "Inactive", "Warning", "Error"]),
        }
        nodes.append(node)

    # Generate links between nodes
    links = []
    for i in range(len(nodes)):
        # Each node connects to 2-5 other nodes
        num_connections = random.randint(2, 5)
        connected_nodes = random.sample([j for j in range(len(nodes)) if j != i], min(num_connections, len(nodes) - 1))

        for j in connected_nodes:
            # Avoid duplicate links
            existing_link = any(
                (link["source"] == f"node_{i}" and link["target"] == f"node_{j}")
                or (link["source"] == f"node_{j}" and link["target"] == f"node_{i}")
                for link in links
            )

            if not existing_link:
                link = {"source": f"node_{i}", "target": f"node_{j}", "weight": random.randint(1, 10)}
                links.append(link)

    return {"nodes": nodes, "links": links}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(generate_network_graph)
    print(run.name)
    print(run.url)
