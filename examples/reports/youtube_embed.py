import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="youtube_embed",
)


@env.task(report=True)
async def generate_youtube_embed():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Science & Technology Video Gallery</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                font-size: 3em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
                background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .subtitle {
                text-align: center;
                font-size: 1.3em;
                margin-bottom: 40px;
                opacity: 0.9;
                font-weight: 300;
            }
            .featured-video {
                margin-bottom: 40px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 20px;
                padding: 30px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }
            .featured-video h2 {
                color: #64c8ff;
                font-size: 2em;
                margin-bottom: 20px;
                text-align: center;
            }
            .main-video {
                position: relative;
                width: 100%;
                max-width: 900px;
                margin: 0 auto;
                aspect-ratio: 16/9;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            }
            .main-video iframe {
                width: 100%;
                height: 100%;
                border: none;
            }
            .video-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 25px;
                margin-top: 40px;
            }
            .video-card {
                background: rgba(255, 255, 255, 0.08);
                border-radius: 15px;
                overflow: hidden;
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
            }
            .video-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
                background: rgba(255, 255, 255, 0.12);
            }
            .video-wrapper {
                position: relative;
                width: 100%;
                aspect-ratio: 16/9;
            }
            .video-wrapper iframe {
                width: 100%;
                height: 100%;
                border: none;
            }
            .video-content {
                padding: 20px;
            }
            .video-title {
                font-size: 1.3em;
                font-weight: 600;
                margin-bottom: 10px;
                color: #fff;
                line-height: 1.3;
            }
            .video-description {
                color: rgba(255, 255, 255, 0.8);
                line-height: 1.5;
                font-size: 0.95em;
            }
            .category-tag {
                display: inline-block;
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 4px 12px;
                border-radius: 15px;
                font-size: 0.8em;
                font-weight: 500;
                margin-bottom: 10px;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 40px 0;
            }
            .stat-card {
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            .stat-number {
                font-size: 2.5em;
                font-weight: bold;
                color: #64c8ff;
                display: block;
                margin-bottom: 5px;
            }
            .stat-label {
                color: rgba(255, 255, 255, 0.8);
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .playlist-section {
                margin-top: 50px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 20px;
                padding: 30px;
                backdrop-filter: blur(10px);
            }
            .playlist-title {
                font-size: 2em;
                color: #64c8ff;
                text-align: center;
                margin-bottom: 30px;
            }
            @media (max-width: 768px) {
                .video-grid {
                    grid-template-columns: 1fr;
                }
                .container {
                    padding: 0 15px;
                }
                h1 {
                    font-size: 2.2em;
                }
                .featured-video {
                    padding: 20px;
                }
            }
            .loading-overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.7);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 1.1em;
                backdrop-filter: blur(2px);
                opacity: 1;
                transition: opacity 0.5s ease;
            }
            .spinner {
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-top: 2px solid #64c8ff;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Science & Technology Video Gallery</h1>
            <p class="subtitle">Curated collection of fascinating scientific and technological content</p>
            
            <!-- Featured Video -->
            <div class="featured-video">
                <h2>🌟 Featured: How SpaceX Lands Rockets</h2>
                <div class="main-video">
                    <div class="loading-overlay">
                        <div class="spinner"></div>
                        Loading video...
                    </div>
                    <iframe src="https://www.youtube.com/embed/4Ca6x4QbpoM?autoplay=0&rel=0&showinfo=0"
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
                             picture-in-picture"
                            allowfullscreen
                            onload="this.previousElementSibling.style.opacity='0'">
                    </iframe>
                </div>
            </div>

            <!-- Stats Section -->
            <div class="stats">
                <div class="stat-card">
                    <span class="stat-number">6</span>
                    <span class="stat-label">Videos Featured</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">∞</span>
                    <span class="stat-label">Learning Opportunities</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">🚀</span>
                    <span class="stat-label">Innovation Level</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">100%</span>
                    <span class="stat-label">Mind-Blowing</span>
                </div>
            </div>

            <!-- Video Grid -->
            <div class="video-grid">
                <div class="video-card">
                    <div class="video-wrapper">
                        <div class="loading-overlay">
                            <div class="spinner"></div>
                            Loading...
                        </div>
                        <iframe src="https://www.youtube.com/embed/7Hlb8Z4H6dM?rel=0"
                                allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowfullscreen
                                onload="this.previousElementSibling.style.opacity='0'">
                        </iframe>
                    </div>
                    <div class="video-content">
                        <div class="category-tag">Quantum Physics</div>
                        <div class="video-title">Quantum Computing Explained</div>
                        <div class="video-description">
                            A comprehensive explanation of quantum computing principles, quantum bits, and how quantum
                             computers could revolutionize computation in the coming decades.
                        </div>
                    </div>
                </div>

                <div class="video-card">
                    <div class="video-wrapper">
                        <div class="loading-overlay">
                            <div class="spinner"></div>
                            Loading...
                        </div>
                        <iframe src="https://www.youtube.com/embed/S_f2qV2_U00?rel=0"
                                allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowfullscreen
                                onload="this.previousElementSibling.style.opacity='0'">
                        </iframe>
                    </div>
                    <div class="video-content">
                        <div class="category-tag">Artificial Intelligence</div>
                        <div class="video-title">The AI Revolution</div>
                        <div class="video-description">
                            Exploring the current state of artificial intelligence, machine learning breakthroughs, and
                             the potential future impact on society and technology.
                        </div>
                    </div>
                </div>

                <div class="video-card">
                    <div class="video-wrapper">
                        <div class="loading-overlay">
                            <div class="spinner"></div>
                            Loading...
                        </div>
                        <iframe src="https://www.youtube.com/embed/mRsT6XkZBWc?rel=0"
                                allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowfullscreen
                                onload="this.previousElementSibling.style.opacity='0'">
                        </iframe>
                    </div>
                    <div class="video-content">
                        <div class="category-tag">Space Exploration</div>
                        <div class="video-title">Journey to Mars</div>
                        <div class="video-description">
                            NASA's ambitious plans for human missions to Mars, including the technologies being
                             developed and challenges that must be overcome.
                        </div>
                    </div>
                </div>

                <div class="video-card">
                    <div class="video-wrapper">
                        <div class="loading-overlay">
                            <div class="spinner"></div>
                            Loading...
                        </div>
                        <iframe src="https://www.youtube.com/embed/wJyUtbn0O5Y?rel=0"
                                allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowfullscreen
                                onload="this.previousElementSibling.style.opacity='0'">
                        </iframe>
                    </div>
                    <div class="video-content">
                        <div class="category-tag">Biology</div>
                        <div class="video-title">CRISPR Gene Editing</div>
                        <div class="video-description">
                            Understanding CRISPR-Cas9 technology, its applications in medicine and biology, and the 
                            ethical considerations surrounding gene editing.
                        </div>
                    </div>
                </div>

                <div class="video-card">
                    <div class="video-wrapper">
                        <div class="loading-overlay">
                            <div class="spinner"></div>
                            Loading...
                        </div>
                        <iframe src="https://www.youtube.com/embed/QwoNP9QWr4Y?rel=0"
                                allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowfullscreen
                                onload="this.previousElementSibling.style.opacity='0'">
                        </iframe>
                    </div>
                    <div class="video-content">
                        <div class="category-tag">Climate Science</div>
                        <div class="video-title">Climate Change Solutions</div>
                        <div class="video-description">
                            Exploring innovative technologies and approaches being developed to combat climate change,
                             from carbon capture to renewable energy advances.
                        </div>
                    </div>
                </div>

                <div class="video-card">
                    <div class="video-wrapper">
                        <div class="loading-overlay">
                            <div class="spinner"></div>
                            Loading...
                        </div>
                        <iframe src="https://www.youtube.com/embed/QsSKBMjuLhE?rel=0"
                                allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowfullscreen
                                onload="this.previousElementSibling.style.opacity='0'">
                        </iframe>
                    </div>
                    <div class="video-content">
                        <div class="category-tag">Neuroscience</div>
                        <div class="video-title">Brain-Computer Interface</div>
                        <div class="video-description">
                            Recent advances in brain-computer interfaces, including Neuralink and other technologies
                            that could allow direct communication between the brain and computers.
                        </div>
                    </div>
                </div>
            </div>

            <!-- Playlist Section -->
            <div class="playlist-section">
                <h2 class="playlist-title">🎬 Complete Science Playlist</h2>
                <div class="main-video">
                    <div class="loading-overlay">
                        <div class="spinner"></div>
                        Loading playlist...
                    </div>
                    <iframe
                     src="https://www.youtube.com/embed/videoseries?list=PLrAXtmRdnEQy5tKjSBBgS7L5MoLcI2a8g&rel=0"
                            allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                            allowfullscreen
                            onload="this.previousElementSibling.style.opacity='0'">
                    </iframe>
                </div>
                <p style="text-align: center; margin-top: 20px; color: rgba(255, 255, 255, 0.8);">
                    A curated playlist featuring the best science and technology educational content
                </p>
            </div>
        </div>

        <script>
            // Add entrance animations
            document.addEventListener('DOMContentLoaded', function() {
                const cards = document.querySelectorAll('.video-card');
                cards.forEach((card, index) => {
                    card.style.opacity = '0';
                    card.style.transform = 'translateY(30px)';
                    setTimeout(() => {
                        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
                        card.style.opacity = '1';
                        card.style.transform = 'translateY(0)';
                    }, index * 100 + 200);
                });

                // Animate stats
                const statNumbers = document.querySelectorAll('.stat-number');
                statNumbers.forEach((stat, index) => {
                    if (stat.textContent !== '∞' && stat.textContent !== '🚀') {
                        const finalValue = parseInt(stat.textContent);
                        if (!isNaN(finalValue)) {
                            animateNumber(stat, 0, finalValue, 1000 + index * 200);
                        }
                    }
                });
            });

            function animateNumber(element, start, end, duration) {
                const startTime = performance.now();
                
                function updateNumber(currentTime) {
                    const elapsed = currentTime - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    const current = Math.floor(progress * (end - start) + start);
                    
                    element.textContent = current + (end === 100 ? '%' : '');
                    
                    if (progress < 1) {
                        requestAnimationFrame(updateNumber);
                    } else {
                        element.textContent = end + (end === 100 ? '%' : '');
                    }
                }
                
                requestAnimationFrame(updateNumber);
            }

            // Handle iframe loading
            document.querySelectorAll('iframe').forEach(iframe => {
                iframe.addEventListener('load', function() {
                    const overlay = this.parentElement.querySelector('.loading-overlay');
                    if (overlay) {
                        overlay.style.opacity = '0';
                        setTimeout(() => overlay.remove(), 500);
                    }
                });
            });
        </script>
    </body>
    </html>
    """

    await flyte.report.replace.aio(html_content)
    await flyte.report.flush.aio()


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(generate_youtube_embed)
    print(run.name)
    print(run.url)
