import flyte
import flyte.report

env = flyte.TaskEnvironment(name="satellite_images", image=flyte.Image.from_debian_base().with_pip_packages("requests"))


@env.task(report=True)
async def generate_satellite_images():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Satellite Images Gallery</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 30px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }
            .gallery {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .image-card {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                backdrop-filter: blur(10px);
            }
            .image-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            }
            .image-card img {
                width: 100%;
                height: 250px;
                object-fit: cover;
                cursor: pointer;
            }
            .card-content {
                padding: 15px;
            }
            .card-title {
                font-size: 1.3em;
                font-weight: bold;
                margin-bottom: 8px;
                color: #fff;
            }
            .card-description {
                color: #e0e0e0;
                line-height: 1.4;
            }
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.9);
                backdrop-filter: blur(5px);
            }
            .modal-content {
                position: relative;
                margin: 5% auto;
                width: 90%;
                max-width: 800px;
                text-align: center;
            }
            .modal img {
                max-width: 100%;
                max-height: 70vh;
                border-radius: 10px;
            }
            .close {
                position: absolute;
                top: -40px;
                right: 0;
                color: #fff;
                font-size: 35px;
                font-weight: bold;
                cursor: pointer;
            }
            .live-feed {
                margin-top: 40px;
                padding: 20px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            .earth-container {
                width: 100%;
                height: 400px;
                border-radius: 10px;
                overflow: hidden;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .svg-image {
                width: 100%;
                height: 250px;
                cursor: pointer;
                overflow: hidden;
            }
            .svg-image svg {
                transition: transform 0.3s ease;
            }
            .svg-image:hover svg {
                transform: scale(1.05);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛰️ Earth from Above: Satellite Imagery Collection</h1>

            <div class="gallery">
                <div class="image-card">
                    <div class="svg-image" onclick="openModal(this)">
                        <svg width="100%" height="250" viewBox="0 0 400 250">
                            <!-- South America Night Lights -->
                            <defs>
                                <radialGradient id="cityGlow" cx="50%" cy="50%" r="50%">
                                    <stop offset="0%" style="stop-color:#ffeb3b;stop-opacity:1" />
                                    <stop offset="100%" style="stop-color:#ff9800;stop-opacity:0.3" />
                                </radialGradient>
                            </defs>
                            <rect width="100%" height="100%" fill="#001122"/>
                            <!-- Continent outline -->
                            <path 
                            d="M 100 50 Q 150 40 180 80 L 190 120 Q 170 180 140 200 Q 120 190 110 160 Q 90 120 100 50"
                                  fill="#002244" stroke="#334455" stroke-width="1"/>
                            <!-- City lights -->
                            <circle cx="150" cy="80" r="3" fill="url(#cityGlow)"/>
                            <circle cx="130" cy="100" r="4" fill="url(#cityGlow)"/>
                            <circle cx="160" cy="110" r="2" fill="url(#cityGlow)"/>
                            <circle cx="140" cy="130" r="3" fill="url(#cityGlow)"/>
                            <circle cx="170" cy="140" r="2" fill="url(#cityGlow)"/>
                            <circle cx="145" cy="160" r="3" fill="url(#cityGlow)"/>
                            <!-- Scattered smaller lights -->
                            <circle cx="120" cy="75" r="1" fill="#ffeb3b" opacity="0.8"/>
                            <circle cx="175" cy="95" r="1" fill="#ffeb3b" opacity="0.6"/>
                            <circle cx="135" cy="145" r="1" fill="#ffeb3b" opacity="0.7"/>
                        </svg>
                    </div>
                    <div class="card-content">
                        <div class="card-title">South America at Night</div>
                        <div class="card-description">Artistic representation of South America's city lights as
                        captured by satellite imagery, showing major urban centers and population density.</div>
                    </div>
                </div>

                <div class="image-card">
                    <div class="svg-image" onclick="openModal(this)">
                        <svg width="100%" height="250" viewBox="0 0 400 250">
                            <!-- Hurricane spiral -->
                            <defs>
                                <radialGradient id="hurricane" cx="50%" cy="50%" r="50%">
                                    <stop offset="0%" style="stop-color:#ffffff;stop-opacity:0.1" />
                                    <stop offset="30%" style="stop-color:#e3f2fd;stop-opacity:0.8" />
                                    <stop offset="100%" style="stop-color:#1565c0;stop-opacity:1" />
                                </radialGradient>
                            </defs>
                            <rect width="100%" height="100%" fill="#0d47a1"/>
                            <!-- Hurricane eye and spiral -->
                            <circle cx="200" cy="125" r="80" fill="url(#hurricane)" opacity="0.9"/>
                            <circle cx="200" cy="125" r="15" fill="#001122" opacity="0.8"/>
                            <!-- Spiral arms -->
                            <path d="M 200 125 Q 250 100 280 130 Q 290 160 270 180 Q 240 190 220 170"
                                  stroke="#bbdefb" stroke-width="3" fill="none" opacity="0.7"/>
                            <path d="M 200 125 Q 150 100 120 130 Q 110 160 130 180 Q 160 190 180 170"
                                  stroke="#bbdefb" stroke-width="3" fill="none" opacity="0.7"/>
                            <!-- Cloud bands -->
                            <ellipse cx="200" cy="125" rx="60" ry="30" fill="#ffffff" opacity="0.3"
                                transform="rotate(30 200 125)"/>
                            <ellipse cx="200" cy="125" rx="70" ry="25" fill="#ffffff" opacity="0.2"
                                transform="rotate(-45 200 125)"/>
                        </svg>
                    </div>
                    <div class="card-content">
                        <div class="card-title">Hurricane Formation</div>
                        <div class="card-description">Stylized representation of a hurricane showing the
                        characteristic spiral structure, eye wall, and cloud bands as seen from satellite imagery.</div>
                    </div>
                </div>

                <div class="image-card">
                    <div class="svg-image" onclick="openModal(this)">
                        <svg width="100%" height="250" viewBox="0 0 400 250">
                            <!-- Amazon rainforest -->
                            <rect width="100%" height="100%" fill="#1b5e20"/>
                            <!-- River systems -->
                            <path d="M 50 125 Q 150 120 250 125 Q 300 130 350 125"
                                  stroke="#4fc3f7" stroke-width="4" fill="none"/>
                            <path d="M 150 100 Q 180 110 200 125"
                                  stroke="#4fc3f7" stroke-width="2" fill="none"/>
                            <path d="M 180 140 Q 220 145 250 125"
                                  stroke="#4fc3f7" stroke-width="2" fill="none"/>
                            <!-- Forest canopy -->
                            <circle cx="100" cy="80" r="20" fill="#2e7d32" opacity="0.8"/>
                            <circle cx="130" cy="75" r="25" fill="#388e3c" opacity="0.9"/>
                            <circle cx="160" cy="85" r="18" fill="#2e7d32" opacity="0.7"/>
                            <circle cx="200" cy="70" r="30" fill="#4caf50" opacity="0.8"/>
                            <circle cx="240" cy="80" r="22" fill="#388e3c" opacity="0.9"/>
                            <circle cx="280" cy="90" r="28" fill="#2e7d32" opacity="0.8"/>
                            <!-- Deforested areas -->
                            <rect x="150" y="160" width="30" height="20" fill="#8d6e63" opacity="0.7"/>
                            <rect x="220" y="170" width="25" height="15" fill="#8d6e63" opacity="0.7"/>
                            <rect x="190" y="180" width="40" height="25" fill="#a1887f" opacity="0.6"/>
                        </svg>
                    </div>
                    <div class="card-content">
                        <div class="card-title">Amazon Rainforest</div>
                        <div class="card-description">Satellite view representation showing the Amazon's dense
                        forest canopy, major river systems, and areas of deforestation.</div>
                    </div>
                </div>

                <div class="image-card">
                    <div class="svg-image" onclick="openModal(this)">
                        <svg width="100%" height="250" viewBox="0 0 400 250">
                            <!-- Antarctic waters -->
                            <rect width="100%" height="100%" fill="#263238"/>
                            <!-- Iceberg -->
                            <polygon points="150,100 250,90 280,120 260,160 120,170 100,130"
                                     fill="#e8f5e8" stroke="#b0bec5" stroke-width="2"/>
                            <!-- Iceberg underwater portion -->
                            <polygon points="140,160 270,150 290,200 110,220"
                                     fill="#c8e6c9" opacity="0.6"/>
                            <!-- Ice cracks and texture -->
                            <line x1="170" y1="100" x2="180" y2="150" stroke="#90a4ae" stroke-width="1"/>
                            <line x1="200" y1="95" x2="210" y2="140" stroke="#90a4ae" stroke-width="1"/>
                            <line x1="230" y1="105" x2="235" y2="135" stroke="#90a4ae" stroke-width="1"/>
                            <!-- Water waves -->
                            <path d="M 50 180 Q 80 175 110 180 Q 140 185 170 180"
                                  stroke="#37474f" stroke-width="2" fill="none" opacity="0.8"/>
                            <path d="M 250 185 Q 280 180 310 185 Q 340 190 370 185"
                                  stroke="#37474f" stroke-width="2" fill="none" opacity="0.8"/>
                        </svg>
                    </div>
                    <div class="card-content">
                        <div class="card-title">Giant Iceberg</div>
                        <div class="card-description">Satellite perspective of a massive iceberg drifting through
                        Antarctic waters, showing both the visible portion and suggested underwater mass.</div>
                    </div>
                </div>

                <div class="image-card">
                    <div class="svg-image" onclick="openModal(this)">
                        <svg width="100%" height="250" viewBox="0 0 400 250">
                            <!-- Desert base -->
                            <rect width="100%" height="100%" fill="#d7ccc8"/>
                            <!-- Dust storm -->
                            <ellipse cx="200" cy="100" rx="150" ry="60" fill="#bcaaa4" opacity="0.8"/>
                            <ellipse cx="220" cy="90" rx="120" ry="40" fill="#a1887f" opacity="0.7"/>
                            <ellipse cx="180" cy="110" rx="100" ry="30" fill="#8d6e63" opacity="0.6"/>
                            <!-- Dust particles -->
                            <circle cx="150" cy="80" r="2" fill="#6d4c41" opacity="0.5"/>
                            <circle cx="250" cy="70" r="1.5" fill="#5d4037" opacity="0.6"/>
                            <circle cx="180" cy="60" r="1" fill="#6d4c41" opacity="0.4"/>
                            <circle cx="270" cy="100" r="2.5" fill="#5d4037" opacity="0.7"/>
                            <circle cx="160" cy="120" r="1.5" fill="#6d4c41" opacity="0.5"/>
                            <!-- Sand dunes -->
                            <ellipse cx="100" cy="200" rx="60" ry="20" fill="#efebe9" opacity="0.8"/>
                            <ellipse cx="300" cy="210" rx="80" ry="25" fill="#efebe9" opacity="0.7"/>
                            <ellipse cx="200" cy="220" rx="70" ry="15" fill="#efebe9" opacity="0.6"/>
                        </svg>
                    </div>
                    <div class="card-content">
                        <div class="card-title">Sahara Dust Storm</div>
                        <div class="card-description">Satellite visualization of a massive dust storm crossing the
                        Sahara Desert, showing the characteristic dust plume and underlying desert terrain.</div>
                    </div>
                </div>

                <div class="image-card">
                    <div class="svg-image" onclick="openModal(this)">
                        <svg width="100%" height="250" viewBox="0 0 400 250">
                            <!-- Ocean background -->
                            <rect width="100%" height="100%" fill="#1565c0"/>
                            <!-- Reef structures -->
                            <ellipse cx="150" cy="150" rx="40" ry="30" fill="#00acc1" opacity="0.7"/>
                            <ellipse cx="200" cy="120" rx="50" ry="35" fill="#26c6da" opacity="0.8"/>
                            <ellipse cx="250" cy="160" rx="45" ry="25" fill="#00acc1" opacity="0.6"/>
                            <!-- Coral formations -->
                            <circle cx="130" cy="140" r="8" fill="#ff7043" opacity="0.9"/>
                            <circle cx="170" cy="130" r="6" fill="#ff5722" opacity="0.8"/>
                            <circle cx="210" cy="145" r="10" fill="#ff7043" opacity="0.9"/>
                            <circle cx="240" cy="135" r="7" fill="#ff5722" opacity="0.8"/>
                            <circle cx="180" cy="170" r="9" fill="#ff6f00" opacity="0.7"/>
                            <!-- Sea grass and marine life -->
                            <path d="M 160 180 Q 165 160 170 180 Q 175 160 180 180"
                                  stroke="#4caf50" stroke-width="2" fill="none" opacity="0.6"/>
                            <path d="M 220 185 Q 225 165 230 185 Q 235 165 240 185"
                                  stroke="#66bb6a" stroke-width="2" fill="none" opacity="0.6"/>
                            <!-- Water depth gradients -->
                            <ellipse cx="300" cy="100" rx="60" ry="40" fill="#4fc3f7" opacity="0.3"/>
                            <ellipse cx="100" cy="80" rx="70" ry="30" fill="#81d4fa" opacity="0.4"/>
                        </svg>
                    </div>
                    <div class="card-content">
                        <div class="card-title">Great Barrier Reef</div>
                        <div class="card-description">Satellite representation of coral reef formations, showing the
                        vibrant coral structures, varying water depths, and marine ecosystems.</div>
                    </div>
                </div>
            </div>

            <div class="live-feed">
                <h2>🌍 Earth Overview</h2>
                <p>Stylized representation of Earth as seen from space:</p>
                <div class="earth-container">
                    <svg width="100%" height="400" viewBox="0 0 400 400">
                        <!-- Space background -->
                        <rect width="100%" height="100%" fill="#000011"/>
                        <!-- Stars -->
                        <circle cx="50" cy="50" r="1" fill="#ffffff" opacity="0.8"/>
                        <circle cx="350" cy="80" r="1.5" fill="#ffffff" opacity="0.6"/>
                        <circle cx="100" cy="30" r="1" fill="#ffffff" opacity="0.9"/>
                        <circle cx="320" cy="40" r="1" fill="#ffffff" opacity="0.7"/>
                        <circle cx="70" cy="350" r="1.5" fill="#ffffff" opacity="0.8"/>
                        <circle cx="330" cy="320" r="1" fill="#ffffff" opacity="0.6"/>

                        <!-- Earth -->
                        <circle cx="200" cy="200" r="120" fill="#1976d2"/>

                        <!-- Continents -->
                        <!-- North America -->
                        <path d="M 140 160 Q 160 150 180 165 Q 190 180 175 190 Q 155 185 140 160"
                              fill="#388e3c" opacity="0.9"/>
                        <!-- Europe/Africa -->
                        <path d="M 200 140 Q 220 135 235 150 Q 240 170 230 185 Q 210 180 200 165 Q 195 150 200 140"
                              fill="#689f38" opacity="0.9"/>
                        <!-- Asia -->
                        <path d="M 240 150 Q 270 145 285 165 Q 280 185 265 180 Q 245 175 240 150"
                              fill="#7cb342" opacity="0.9"/>
                        <!-- South America -->
                        <path d="M 160 210 Q 175 205 180 225 Q 175 245 160 240 Q 150 225 160 210"
                              fill="#8bc34a" opacity="0.9"/>
                        <!-- Australia -->
                        <path d="M 250 230 Q 265 225 270 235 Q 265 245 250 240 Q 245 235 250 230"
                              fill="#9ccc65" opacity="0.9"/>
                        <!-- Clouds -->
                        <ellipse cx="170" cy="180" rx="30" ry="15" fill="#ffffff" opacity="0.7"
                         transform="rotate(20 170 180)"/>
                        <ellipse cx="240" cy="200" rx="25" ry="12" fill="#ffffff" opacity="0.6"
                         transform="rotate(-15 240 200)"/>
                        <ellipse cx="200" cy="240" rx="35" ry="18" fill="#ffffff" opacity="0.8" 
                        transform="rotate(30 200 240)"/>
                        
                        <!-- Atmosphere glow -->
                        <circle cx="200" cy="200" r="125" fill="none" stroke="#81d4fa" stroke-width="3" opacity="0.6"/>
                        <circle cx="200" cy="200" r="130" fill="none" stroke="#e3f2fd" stroke-width="2" opacity="0.3"/>
                    </svg>
                </div>
            </div>
        </div>

        <!-- Modal for enlarged images -->
        <div id="imageModal" class="modal" onclick="closeModal()">
            <div class="modal-content">
                <span class="close" onclick="closeModal()">&times;</span>
                <div id="modalImage"></div>
            </div>
        </div>

        <script>
            function openModal(element) {
                const modal = document.getElementById('imageModal');
                const modalImg = document.getElementById('modalImage');
                modal.style.display = 'block';
                
                // Clone the SVG content for the modal
                const svgContent = element.innerHTML;
                modalImg.innerHTML = svgContent;
                
                // Scale up the SVG in the modal
                const svg = modalImg.querySelector('svg');
                if (svg) {
                    svg.style.width = '100%';
                    svg.style.height = 'auto';
                    svg.style.maxHeight = '70vh';
                }
            }

            function closeModal() {
                document.getElementById('imageModal').style.display = 'none';
            }

            // Add loading animation
            document.addEventListener('DOMContentLoaded', function() {
                const cards = document.querySelectorAll('.image-card');
                cards.forEach((card, index) => {
                    card.style.opacity = '0';
                    card.style.transform = 'translateY(20px)';
                    setTimeout(() => {
                        card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                        card.style.opacity = '1';
                        card.style.transform = 'translateY(0)';
                    }, index * 100);
                });
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
    run = flyte.run(generate_satellite_images)
    print(run.name)
    print(run.url)
