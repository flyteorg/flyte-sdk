import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="protein_3d",
)


@env.task(report=True)
async def generate_protein_3d():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>3D Protein Structure Viewer</title>
        <!-- Self-contained CSS-only 3D protein visualization -->
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }
            .subtitle {
                text-align: center;
                font-size: 1.2em;
                margin-bottom: 30px;
                opacity: 0.9;
            }
            .main-viewer {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }
            .viewer-container {
                display: flex;
                gap: 20px;
                align-items: stretch;
            }
            .protein-display {
                flex: 2;
                height: 500px;
                border: 2px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                background: #000;
                position: relative;
                overflow: hidden;
            }
            .protein-structure {
                display: none;
                width: 100%;
                height: 100%;
                position: absolute;
                top: 0;
                left: 0;
            }
            .protein-structure.active {
                display: block;
            }
            .protein-visual {
                width: 100%;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                animation: rotate 20s linear infinite;
            }
            @keyframes rotate {
                from { transform: rotateY(0deg); }
                to { transform: rotateY(360deg); }
            }
            .protein-visual.paused {
                animation-play-state: paused;
            }
            .controls {
                flex: 1;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 20px;
                backdrop-filter: blur(5px);
            }
            .control-group {
                margin-bottom: 20px;
            }
            .control-group h3 {
                margin: 0 0 10px 0;
                font-size: 1.1em;
                color: #fff;
            }
            .button-group {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 10px;
            }
            button {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-size: 0.9em;
            }
            button:hover {
                background: rgba(255, 255, 255, 0.3);
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            }
            button.active {
                background: rgba(100, 200, 255, 0.6);
                border-color: rgba(100, 200, 255, 0.8);
            }
            .protein-info {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .info-card {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 20px;
                backdrop-filter: blur(10px);
            }
            .info-card h3 {
                margin-top: 0;
                color: #64c8ff;
                font-size: 1.3em;
            }
            .info-card p {
                line-height: 1.5;
                margin-bottom: 10px;
            }
            .loading {
                text-align: center;
                padding: 40px;
                font-size: 1.2em;
            }
            .spinner {
                border: 3px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top: 3px solid #64c8ff;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧬 3D Protein Structure Viewer</h1>
            <p class="subtitle">Interactive visualization of protein structures from the Protein Data Bank</p>

            <div class="main-viewer">
                <div class="viewer-container">
                    <div id="proteinViewer" class="protein-display">
                        <div class="protein-structure active" data-protein="ubiquitin">
                            <div class="protein-visual">
                                <svg width="100%" height="400" viewBox="0 0 400 400">
                                    <!-- Simplified 3D representation of Ubiquitin -->
                                    <defs>
                                        <linearGradient id="helixGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                            <stop offset="0%" style="stop-color:#ff6b6b;stop-opacity:1" />
                                            <stop offset="100%" style="stop-color:#ee5a52;stop-opacity:1" />
                                        </linearGradient>
                                        <linearGradient id="sheetGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                            <stop offset="0%" style="stop-color:#4ecdc4;stop-opacity:1" />
                                            <stop offset="100%" style="stop-color:#45b7d1;stop-opacity:1" />
                                        </linearGradient>
                                    </defs>
                                    <!-- Alpha helix -->
                                    <ellipse cx="150" cy="100" rx="80" ry="15" fill="url(#helixGrad)"
                                        transform="rotate(30 150 100)"/>
                                    <ellipse cx="155" cy="105" rx="80" ry="12" fill="url(#helixGrad)"
                                        transform="rotate(30 155 105)" opacity="0.8"/>
                                    <ellipse cx="160" cy="110" rx="80" ry="10" fill="url(#helixGrad)"
                                        transform="rotate(30 160 110)" opacity="0.6"/>

                                    <!-- Beta sheets -->
                                    <rect x="100" y="200" width="100" height="20" fill="url(#sheetGrad)"
                                        transform="rotate(-15 150 210)"/>
                                    <rect x="105" y="225" width="100" height="20" fill="url(#sheetGrad)"
                                        transform="rotate(-15 155 235)" opacity="0.8"/>
                                    <rect x="110" y="250" width="100" height="20" fill="url(#sheetGrad)"
                                        transform="rotate(-15 160 260)" opacity="0.6"/>
                                    <!-- Random coils -->
                                    <path d="M 250 150 Q 280 120 300 180 Q 320 240 280 270 Q 240 280 220 250"
                                          stroke="#f9ca24" stroke-width="8" fill="none" opacity="0.8"/>
                                    <!-- Key residues -->
                                    <circle cx="180" cy="160" r="8" fill="#6c5ce7" opacity="0.9">
                                        <title>Lys48 - Ubiquitin linkage site</title>
                                    </circle>
                                    <circle cx="220" cy="180" r="8" fill="#a29bfe" opacity="0.9">
                                        <title>Lys63 - Alternative linkage site</title>
                                    </circle>
                                </svg>
                            </div>
                        </div>

                        <div class="protein-structure" data-protein="lysozyme">
                            <div class="protein-visual">
                                <svg width="100%" height="400" viewBox="0 0 400 400">
                                    <!-- Simplified 3D representation of Lysozyme -->
                                    <defs>
                                        <radialGradient id="enzymeGrad">
                                            <stop offset="0%" style="stop-color:#74b9ff;stop-opacity:1" />
                                            <stop offset="100%" style="stop-color:#0984e3;stop-opacity:1" />
                                        </radialGradient>
                                    </defs>

                                    <!-- Main globular structure -->
                                    <ellipse cx="200" cy="200" rx="120" ry="100" fill="url(#enzymeGrad)" opacity="0.7"/>

                                    <!-- Alpha helices -->
                                    <rect x="120" y="150" width="60" height="12" rx="6" fill="#ff6b6b"
                                        transform="rotate(45 150 156)"/>
                                    <rect x="220" y="180" width="70" height="12" rx="6" fill="#ff6b6b"
                                        transform="rotate(-30 255 186)"/>
                                    <rect x="160" y="250" width="80" height="12" rx="6" fill="#ff6b6b"
                                        transform="rotate(15 200 256)"/>

                                    <!-- Beta sheets -->
                                    <polygon points="140,220 180,210 185,225 145,235" fill="#4ecdc4" opacity="0.8"/>
                                    <polygon points="250,160 290,150 295,165 255,175" fill="#4ecdc4" opacity="0.8"/>

                                    <!-- Active site cleft -->
                                    <path d="M 180 180 Q 200 160 220 180 Q 240 200 220 220 Q 200 240 180 220 Q 160 200 180 180"
                                          fill="#2d3436" opacity="0.6"/>
                                    <!-- Disulfide bonds -->
                                    <line x1="160" y1="170" x2="240" y2="190" stroke="#fdcb6e"
                                        stroke-width="3" opacity="0.8"/>
                                    <line x1="170" y1="230" x2="230" y2="240" stroke="#fdcb6e"
                                        stroke-width="3" opacity="0.8"/>
                                </svg>
                            </div>
                        </div>
                    </div>

                    <div class="controls">
                        <div class="control-group">
                            <h3>Protein Selection</h3>
                            <div class="button-group">
                                <button onclick="showProtein('ubiquitin')" class="active"
                                    data-protein="ubiquitin">Ubiquitin</button>
                                <button onclick="showProtein('lysozyme')" data-protein="lysozyme">Lysozyme</button>
                            </div>
                        </div>

                        <div class="control-group">
                            <h3>View Controls</h3>
                            <div class="button-group">
                                <button onclick="rotateProtein()">Rotate View</button>
                                <button onclick="highlightFeatures()">Show Features</button>
                                <button onclick="resetView()">Reset</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="protein-info">
                <div class="info-card" id="proteinInfo">
                    <h3 id="proteinTitle">Current Protein: Ubiquitin</h3>
                    <div id="proteinDescription">
                        <p><strong>Function:</strong> Small regulatory protein that tags other proteins
                        for degradation by the proteasome.</p>
                        <p><strong>Size:</strong> 76 amino acids</p>
                        <p><strong>Structure:</strong> Compact globular protein with a β-grasp fold</p>
                        <p><strong>Key Features:</strong> Two critical lysine residues (Lys48, Lys63)
                        for different types of ubiquitin linkages</p>
                    </div>
                </div>

                <div class="info-card">
                    <h3>Structure Elements</h3>
                    <p><strong>α-Helices (Red):</strong> Regular secondary structures with hydrogen bonding patterns</p>
                    <p><strong>β-Sheets (Blue):</strong> Extended conformations forming flat, rigid structures</p>
                    <p><strong>Random Coils (Yellow):</strong> Flexible loops connecting structured regions</p>
                    <p><strong>Active Sites (Purple):</strong> Functional regions critical for protein activity</p>
                </div>

                <div class="info-card">
                    <h3>Visualization Features</h3>
                    <p><strong>Simplified 3D View:</strong> Shows major structural elements and their
                    spatial relationships</p>
                    <p><strong>Color Coding:</strong> Different colors represent different types of
                    secondary structures</p>
                    <p><strong>Interactive Elements:</strong> Hover over features to see detailed information</p>
                    <p><strong>Comparative View:</strong> Switch between different protein structures to compare</p>
                </div>
            </div>
        </div>

        <script>
            let currentProtein = 'ubiquitin';
            let isRotating = true;

            const proteinData = {
                ubiquitin: {
                    name: 'Ubiquitin',
                    function: 'Small regulatory protein that tags other proteins for degradation' +
                        ' by the proteasome.',
                    size: '76 amino acids',
                    structure: 'Compact globular protein with a β-grasp fold',
                    features: 'Two critical lysine residues (Lys48, Lys63) for different types of' +
                        ' ubiquitin linkages'
                },
                lysozyme: {
                    name: 'Lysozyme',
                    function: 'Antimicrobial enzyme that breaks down bacterial cell walls by' +
                        ' cleaving peptidoglycan.',
                    size: '129 amino acids',
                    structure: 'Compact enzyme with deep active site cleft and disulfide bonds',
                    features: 'Active site cleft for substrate binding and multiple disulfide' +
                        ' bridges for stability'
                }
            };

            function showProtein(proteinId) {
                currentProtein = proteinId;

                // Hide all protein structures
                document.querySelectorAll('.protein-structure').forEach(struct => {
                    struct.classList.remove('active');
                });

                // Show selected protein
                document.querySelector(`[data-protein="${proteinId}"]`).classList.add('active');

                // Update active button
                document.querySelectorAll('[data-protein]').forEach(btn => {
                    btn.classList.remove('active');
                });
                document.querySelector(`button[data-protein="${proteinId}"]`).classList.add('active');

                // Update protein info
                updateProteinInfo(proteinId);
            }

            function updateProteinInfo(proteinId) {
                const info = proteinData[proteinId];
                if (info) {
                    document.getElementById('proteinTitle').textContent = `Current Protein: ${info.name}`;
                    document.getElementById('proteinDescription').innerHTML = `
                        <p><strong>Function:</strong> ${info.function}</p>
                        <p><strong>Size:</strong> ${info.size}</p>
                        <p><strong>Structure:</strong> ${info.structure}</p>
                        <p><strong>Key Features:</strong> ${info.features}</p>
                    `;
                }
            }

            function rotateProtein() {
                isRotating = !isRotating;
                const visual = document.querySelector('.protein-structure.active .protein-visual');
                if (isRotating) {
                    visual.classList.remove('paused');
                } else {
                    visual.classList.add('paused');
                }
            }

            function highlightFeatures() {
                const activeProtein = document.querySelector('.protein-structure.active');
                const features = activeProtein.querySelectorAll('circle, path[stroke="#fdcb6e"]');
                features.forEach((feature, index) => {
                    setTimeout(() => {
                        feature.style.filter = 'brightness(2) drop-shadow(0 0 10px currentColor)';
                        setTimeout(() => {
                            feature.style.filter = '';
                        }, 1000);
                    }, index * 500);
                });
            }

            function resetView() {
                const visual = document.querySelector('.protein-structure.active .protein-visual');
                visual.style.transform = '';
                visual.classList.remove('paused');
                isRotating = true;
            }

            // Initialize
            document.addEventListener('DOMContentLoaded', function() {
                updateProteinInfo(currentProtein);
            });
        </script>
    </body>
    </html>
    """

    await flyte.report.replace.aio(html_content)
    await flyte.report.flush.aio()


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    run = flyte.run(generate_protein_3d)
    print(run.name)
    print(run.url)
