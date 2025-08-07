import json
import random

import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="globe_visualization",
)


@env.task(report=True)
async def generate_globe_visualization():
    # Generate sample data points around the globe
    data_points = generate_globe_data()

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interactive 3D Globe Visualization</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: radial-gradient(ellipse at center, #0a0a0a 0%, #000000 100%);
                color: white;
                overflow: hidden;
                height: 100vh;
            }}
            #globeContainer {{
                position: relative;
                width: 100vw;
                height: 100vh;
            }}
            .ui-overlay {{
                position: absolute;
                top: 0;
                left: 0;
                z-index: 100;
                pointer-events: none;
            }}
            .header {{
                position: absolute;
                top: 20px;
                left: 20px;
                pointer-events: none;
            }}
            .header h1 {{
                font-size: 2.5em;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
                background: linear-gradient(45deg, #64b5f6, #42a5f5, #2196f3);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .header p {{
                font-size: 1.1em;
                margin: 5px 0 0 0;
                opacity: 0.8;
            }}
            .controls {{
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(0, 0, 0, 0.7);
                border-radius: 10px;
                padding: 15px;
                pointer-events: auto;
                backdrop-filter: blur(10px);
                min-width: 200px;
            }}
            .control-group {{
                margin-bottom: 15px;
            }}
            .control-group:last-child {{
                margin-bottom: 0;
            }}
            .control-group label {{
                display: block;
                font-size: 0.9em;
                margin-bottom: 5px;
                color: #ccc;
            }}
            .control-group select, .control-group input {{
                width: 100%;
                padding: 6px;
                border: 1px solid #555;
                border-radius: 4px;
                background: rgba(255, 255, 255, 0.1);
                color: white;
                font-size: 0.9em;
            }}
            .control-group button {{
                width: 100%;
                padding: 8px;
                border: none;
                border-radius: 4px;
                background: linear-gradient(45deg, #2196f3, #21cbf3);
                color: white;
                cursor: pointer;
                font-size: 0.9em;
                transition: all 0.3s ease;
            }}
            .control-group button:hover {{
                background: linear-gradient(45deg, #1976d2, #0288d1);
                transform: translateY(-1px);
            }}
            .info-panel {{
                position: absolute;
                bottom: 20px;
                left: 20px;
                background: rgba(0, 0, 0, 0.8);
                border-radius: 10px;
                padding: 20px;
                max-width: 350px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                pointer-events: auto;
            }}
            .data-point-info {{
                display: none;
            }}
            .data-point-info.active {{
                display: block;
            }}
            .data-point-info h3 {{
                margin: 0 0 10px 0;
                color: #64b5f6;
                font-size: 1.2em;
            }}
            .data-point-info .metric {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
                padding: 4px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }}
            .data-point-info .metric:last-child {{
                border-bottom: none;
                margin-bottom: 0;
            }}
            .metric-label {{
                color: #ccc;
                font-size: 0.9em;
            }}
            .metric-value {{
                color: #fff;
                font-weight: bold;
                font-size: 0.9em;
            }}
            .legend {{
                position: absolute;
                bottom: 20px;
                right: 20px;
                background: rgba(0, 0, 0, 0.8);
                border-radius: 10px;
                padding: 15px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            .legend h4 {{
                margin: 0 0 10px 0;
                color: #64b5f6;
                font-size: 1em;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin-bottom: 6px;
                font-size: 0.8em;
            }}
            .legend-color {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }}
            .stats {{
                position: absolute;
                top: 50%;
                left: 20px;
                transform: translateY(-50%);
                background: rgba(0, 0, 0, 0.7);
                border-radius: 10px;
                padding: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                min-width: 150px;
            }}
            .stat-item {{
                text-align: center;
                margin-bottom: 15px;
                padding: 10px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
            }}
            .stat-item:last-child {{
                margin-bottom: 0;
            }}
            .stat-number {{
                font-size: 1.8em;
                font-weight: bold;
                color: #64b5f6;
                display: block;
            }}
            .stat-label {{
                font-size: 0.8em;
                color: #ccc;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-top: 2px;
            }}
            .loading {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
                z-index: 200;
            }}
            .loading.hidden {{
                display: none;
            }}
            .spinner {{
                border: 3px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top: 3px solid #64b5f6;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            @media (max-width: 768px) {{
                .controls, .stats, .info-panel, .legend {{
                    display: none;
                }}
                .header h1 {{
                    font-size: 1.8em;
                }}
            }}
        </style>
    </head>
    <body>
        <div id="globeContainer">
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Loading 3D Globe...</p>
            </div>

            <div class="ui-overlay">
                <div class="header">
                    <h1>🌍 Interactive Globe</h1>
                    <p>Global data visualization in 3D space</p>
                </div>

                <div class="controls">
                    <div class="control-group">
                        <label>Visualization Mode:</label>
                        <select id="viewMode" onchange="changeViewMode()">
                            <option value="data">Data Points</option>
                            <option value="connections">Connections</option>
                            <option value="heatmap">Heat Map</option>
                            <option value="all">All Combined</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Animation:</label>
                        <button onclick="toggleRotation()" id="rotateBtn">⏸️ Pause Rotation</button>
                    </div>
                    <div class="control-group">
                        <label>Camera:</label>
                        <button onclick="resetCamera()">🎯 Reset View</button>
                    </div>
                    <div class="control-group">
                        <label>Data Filter:</label>
                        <select id="dataFilter" onchange="filterData()">
                            <option value="all">All Data</option>
                            <option value="high">High Values Only</option>
                            <option value="medium">Medium Values</option>
                            <option value="low">Low Values</option>
                        </select>
                    </div>
                </div>

                <div class="stats">
                    <div class="stat-item">
                        <span class="stat-number" id="totalPoints">0</span>
                        <span class="stat-label">Data Points</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number" id="totalConnections">0</span>
                        <span class="stat-label">Connections</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number" id="totalCountries">0</span>
                        <span class="stat-label">Countries</span>
                    </div>
                </div>

                <div class="info-panel">
                    <div id="defaultInfo">
                        <h3>🌐 Global Data Visualization</h3>
                        <p>This interactive 3D globe displays data points from around the world.
                        Each point represents a location with associated metrics.</p>
                        <p><strong>Controls:</strong></p>
                        <ul>
                            <li>Mouse drag: Rotate globe</li>
                            <li>Mouse wheel: Zoom in/out</li>
                            <li>Click points: View details</li>
                        </ul>
                    </div>
                    <div class="data-point-info" id="pointInfo">
                        <h3 id="pointTitle">Location</h3>
                        <div class="metric">
                            <span class="metric-label">Country:</span>
                            <span class="metric-value" id="pointCountry">-</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Value:</span>
                            <span class="metric-value" id="pointValue">-</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Category:</span>
                            <span class="metric-value" id="pointCategory">-</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Coordinates:</span>
                            <span class="metric-value" id="pointCoords">-</span>
                        </div>
                    </div>
                </div>

                <div class="legend">
                    <h4>Data Categories</h4>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ff4444;"></div>
                        <span>High Priority</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ffaa00;"></div>
                        <span>Medium Priority</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #44ff44;"></div>
                        <span>Low Priority</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #4444ff;"></div>
                        <span>Special</span>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Global variables
            let scene, camera, renderer, globe, controls;
            let dataPoints = {json.dumps(data_points)};
            let pointMeshes = [];
            let connectionLines = [];
            let isRotating = true;
            let currentViewMode = 'data';
            let raycaster, mouse;
            let selectedPoint = null;

            // Initialize the globe
            function init() {{
                // Create scene
                scene = new THREE.Scene();

                // Create camera
                camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                camera.position.z = 250;

                // Create renderer
                renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setClearColor(0x000000, 0);
                document.getElementById('globeContainer').appendChild(renderer.domElement);

                // Create controls
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.enablePan = false;
                controls.minDistance = 150;
                controls.maxDistance = 500;

                // Create raycaster for mouse interactions
                raycaster = new THREE.Raycaster();
                mouse = new THREE.Vector2();

                // Create globe
                createGlobe();

                // Create data points
                createDataPoints();

                // Create connections
                createConnections();

                // Add lights
                addLights();

                // Add event listeners
                addEventListeners();

                // Update UI
                updateStats();

                // Hide loading screen
                document.getElementById('loading').classList.add('hidden');

                // Start animation loop
                animate();
            }}

            function createGlobe() {{
                // Create globe geometry and material
                const geometry = new THREE.SphereGeometry(100, 64, 64);

                // Load earth texture (using a simple color for now)
                const material = new THREE.MeshPhongMaterial({{
                    color: 0x2c5aa0,
                    transparent: true,
                    opacity: 0.8,
                    wireframe: false
                }});

                globe = new THREE.Mesh(geometry, material);
                scene.add(globe);

                // Add atmosphere effect
                const atmosphereGeometry = new THREE.SphereGeometry(102, 64, 64);
                const atmosphereMaterial = new THREE.MeshBasicMaterial({{
                    color: 0x4fc3f7,
                    transparent: true,
                    opacity: 0.1,
                    side: THREE.BackSide
                }});
                const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
                scene.add(atmosphere);

                // Add wireframe overlay
                const wireframeGeometry = new THREE.SphereGeometry(101, 32, 32);
                const wireframeMaterial = new THREE.MeshBasicMaterial({{
                    color: 0x81d4fa,
                    transparent: true,
                    opacity: 0.2,
                    wireframe: true
                }});
                const wireframe = new THREE.Mesh(wireframeGeometry, wireframeMaterial);
                scene.add(wireframe);
            }}

            function createDataPoints() {{
                pointMeshes = [];
                dataPoints.forEach((point, index) => {{
                    const phi = (90 - point.lat) * (Math.PI / 180);
                    const theta = (point.lng + 180) * (Math.PI / 180);

                    const x = -100 * Math.sin(phi) * Math.cos(theta);
                    const y = 100 * Math.cos(phi);
                    const z = 100 * Math.sin(phi) * Math.sin(theta);

                    // Create point geometry
                    const geometry = new THREE.SphereGeometry(point.value / 10, 8, 8);

                    // Determine color based on category
                    let color;
                    switch(point.category) {{
                        case 'high': color = 0xff4444; break;
                        case 'medium': color = 0xffaa00; break;
                        case 'low': color = 0x44ff44; break;
                        case 'special': color = 0x4444ff; break;
                        default: color = 0xffffff;
                    }}

                    const material = new THREE.MeshBasicMaterial({{
                        color: color,
                        transparent: true,
                        opacity: 0.8
                    }});

                    const pointMesh = new THREE.Mesh(geometry, material);
                    pointMesh.position.set(x, y, z);
                    pointMesh.userData = {{ ...point, index }};

                    // Add glow effect
                    const glowGeometry = new THREE.SphereGeometry(point.value / 8, 8, 8);
                    const glowMaterial = new THREE.MeshBasicMaterial({{
                        color: color,
                        transparent: true,
                        opacity: 0.3
                    }});
                    const glow = new THREE.Mesh(glowGeometry, glowMaterial);
                    glow.position.copy(pointMesh.position);
                    glow.scale.multiplyScalar(1.5);

                    scene.add(pointMesh);
                    scene.add(glow);
                    pointMeshes.push(pointMesh);
                }});
            }}

            function createConnections() {{
                connectionLines = [];
                // Create connections between nearby points
                for (let i = 0; i < dataPoints.length; i++) {{
                    for (let j = i + 1; j < dataPoints.length; j++) {{
                        const point1 = dataPoints[i];
                        const point2 = dataPoints[j];

                        // Calculate distance
                        const distance = Math.sqrt(
                            Math.pow(point1.lat - point2.lat, 2) +
                            Math.pow(point1.lng - point2.lng, 2)
                        );

                        // Only connect nearby points
                        if (distance < 50 && Math.random() > 0.7) {{
                            const phi1 = (90 - point1.lat) * (Math.PI / 180);
                            const theta1 = (point1.lng + 180) * (Math.PI / 180);
                            const phi2 = (90 - point2.lat) * (Math.PI / 180);
                            const theta2 = (point2.lng + 180) * (Math.PI / 180);

                            const x1 = -100 * Math.sin(phi1) * Math.cos(theta1);
                            const y1 = 100 * Math.cos(phi1);
                            const z1 = 100 * Math.sin(phi1) * Math.sin(theta1);

                            const x2 = -100 * Math.sin(phi2) * Math.cos(theta2);
                            const y2 = 100 * Math.cos(phi2);
                            const z2 = 100 * Math.sin(phi2) * Math.sin(theta2);

                            const geometry = new THREE.BufferGeometry().setFromPoints([
                                new THREE.Vector3(x1, y1, z1),
                                new THREE.Vector3(x2, y2, z2)
                            ]);

                            const material = new THREE.LineBasicMaterial({{
                                color: 0x64b5f6,
                                transparent: true,
                                opacity: 0.3
                            }});

                            const line = new THREE.Line(geometry, material);
                            scene.add(line);
                            connectionLines.push(line);
                        }}
                    }}
                }}
            }}

            function addLights() {{
                // Ambient light
                const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
                scene.add(ambientLight);

                // Point light
                const pointLight = new THREE.PointLight(0xffffff, 1, 300);
                pointLight.position.set(200, 200, 200);
                scene.add(pointLight);

                // Directional light
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(-200, 200, 200);
                scene.add(directionalLight);
            }}

            function addEventListeners() {{
                window.addEventListener('resize', onWindowResize, false);
                renderer.domElement.addEventListener('click', onMouseClick, false);
            }}

            function onWindowResize() {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }}

            function onMouseClick(event) {{
                mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
                mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(pointMeshes);

                if (intersects.length > 0) {{
                    const selectedObject = intersects[0].object;
                    showPointInfo(selectedObject.userData);
                }} else {{
                    hidePointInfo();
                }}
            }}

            function showPointInfo(pointData) {{
                document.getElementById('defaultInfo').style.display = 'none';
                document.getElementById('pointInfo').classList.add('active');

                document.getElementById('pointTitle').textContent = pointData.city;
                document.getElementById('pointCountry').textContent = pointData.country;
                document.getElementById('pointValue').textContent = pointData.value;
                document.getElementById('pointCategory').textContent = pointData.category.toUpperCase();
                document.getElementById('pointCoords').textContent =
                    `${{pointData.lat.toFixed(2)}}, ${{pointData.lng.toFixed(2)}}`;
            }}

            function hidePointInfo() {{
                document.getElementById('defaultInfo').style.display = 'block';
                document.getElementById('pointInfo').classList.remove('active');
            }}

            function animate() {{
                requestAnimationFrame(animate);

                if (isRotating) {{
                    globe.rotation.y += 0.005;
                }}

                // Animate point glow
                pointMeshes.forEach((point, index) => {{
                    const time = Date.now() * 0.001 + index;
                    point.material.opacity = 0.6 + 0.2 * Math.sin(time);
                }});

                controls.update();
                renderer.render(scene, camera);
            }}

            // Control functions
            function toggleRotation() {{
                isRotating = !isRotating;
                const btn = document.getElementById('rotateBtn');
                btn.textContent = isRotating ? '⏸️ Pause Rotation' : '▶️ Start Rotation';
            }}

            function resetCamera() {{
                camera.position.set(0, 0, 250);
                controls.reset();
            }}

            function changeViewMode() {{
                const mode = document.getElementById('viewMode').value;
                currentViewMode = mode;

                // Hide all elements first
                pointMeshes.forEach(mesh => mesh.visible = false);
                connectionLines.forEach(line => line.visible = false);

                // Show based on mode
                switch(mode) {{
                    case 'data':
                        pointMeshes.forEach(mesh => mesh.visible = true);
                        break;
                    case 'connections':
                        connectionLines.forEach(line => line.visible = true);
                        break;
                    case 'heatmap':
                        pointMeshes.forEach(mesh => {{
                            mesh.visible = true;
                            mesh.scale.set(2, 2, 2);
                        }});
                        break;
                    case 'all':
                        pointMeshes.forEach(mesh => mesh.visible = true);
                        connectionLines.forEach(line => line.visible = true);
                        break;
                }}
            }}

            function filterData() {{
                const filter = document.getElementById('dataFilter').value;

                pointMeshes.forEach(mesh => {{
                    if (filter === 'all') {{
                        mesh.visible = true;
                    }} else {{
                        mesh.visible = mesh.userData.category === filter;
                    }}
                }});

                updateStats();
            }}

            function updateStats() {{
                const visiblePoints = pointMeshes.filter(mesh => mesh.visible).length;
                const visibleConnections = connectionLines.filter(line => line.visible).length;
                const uniqueCountries = [...new Set(dataPoints.map(point => point.country))].length;

                document.getElementById('totalPoints').textContent = visiblePoints;
                document.getElementById('totalConnections').textContent = visibleConnections;
                document.getElementById('totalCountries').textContent = uniqueCountries;
            }}

            // Start the application
            document.addEventListener('DOMContentLoaded', init);
        </script>
    </body>
    </html>
    """

    await flyte.report.replace.aio(html_content)
    await flyte.report.flush.aio()


def generate_globe_data():
    """Generate sample data points for the globe"""
    cities = [
        {"city": "New York", "country": "USA", "lat": 40.7128, "lng": -74.0060},
        {"city": "London", "country": "UK", "lat": 51.5074, "lng": -0.1278},
        {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "lng": 139.6503},
        {"city": "Sydney", "country": "Australia", "lat": -33.8688, "lng": 151.2093},
        {"city": "Paris", "country": "France", "lat": 48.8566, "lng": 2.3522},
        {"city": "São Paulo", "country": "Brazil", "lat": -23.5505, "lng": -46.6333},
        {"city": "Mumbai", "country": "India", "lat": 19.0760, "lng": 72.8777},
        {"city": "Cairo", "country": "Egypt", "lat": 30.0444, "lng": 31.2357},
        {"city": "Moscow", "country": "Russia", "lat": 55.7558, "lng": 37.6176},
        {"city": "Beijing", "country": "China", "lat": 39.9042, "lng": 116.4074},
        {"city": "Lagos", "country": "Nigeria", "lat": 6.5244, "lng": 3.3792},
        {"city": "Mexico City", "country": "Mexico", "lat": 19.4326, "lng": -99.1332},
        {"city": "Bangkok", "country": "Thailand", "lat": 13.7563, "lng": 100.5018},
        {"city": "Istanbul", "country": "Turkey", "lat": 41.0082, "lng": 28.9784},
        {"city": "Buenos Aires", "country": "Argentina", "lat": -34.6118, "lng": -58.3960},
        {"city": "Cape Town", "country": "South Africa", "lat": -33.9249, "lng": 18.4241},
        {"city": "Dubai", "country": "UAE", "lat": 25.2048, "lng": 55.2708},
        {"city": "Singapore", "country": "Singapore", "lat": 1.3521, "lng": 103.8198},
        {"city": "Stockholm", "country": "Sweden", "lat": 59.3293, "lng": 18.0686},
        {"city": "Vancouver", "country": "Canada", "lat": 49.2827, "lng": -123.1207},
    ]

    categories = ["high", "medium", "low", "special"]

    data_points = []
    for city in cities:
        data_point = {**city, "value": random.randint(10, 100), "category": random.choice(categories)}
        data_points.append(data_point)

    return data_points


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(generate_globe_visualization)
    print(run.name)
    print(run.url)
