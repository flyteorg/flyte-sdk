import datetime
import json
import random

import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="interactive_dashboard",
)


@env.task(report=True)
async def generate_interactive_dashboard():
    # Generate sample data
    sample_data = generate_sample_data()

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interactive Analytics Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                min-height: 100vh;
            }}
            .dashboard {{
                padding: 20px;
                max-width: 1600px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }}
            .header h1 {{
                font-size: 3em;
                font-weight: 700;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                margin-bottom: 10px;
            }}
            .header p {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            .controls {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 25px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }}
            .control-group {{
                display: flex;
                gap: 15px;
                align-items: center;
                flex-wrap: wrap;
            }}
            .control-group label {{
                font-weight: 600;
                color: #555;
            }}
            .control-group select, .control-group input {{
                padding: 8px 12px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 14px;
                transition: border-color 0.3s ease;
            }}
            .control-group select:focus, .control-group input:focus {{
                outline: none;
                border-color: #667eea;
            }}
            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 25px;
                margin-bottom: 25px;
            }}
            .chart-container {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 25px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }}
            .chart-container:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
            }}
            .chart-title {{
                font-size: 1.4em;
                font-weight: 600;
                color: #333;
                margin-bottom: 15px;
                text-align: center;
            }}
            .chart-wrapper {{
                position: relative;
                height: 300px;
            }}
            .metric-cards {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 25px;
            }}
            .metric-card {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                backdrop-filter: blur(10px);
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }}
            .metric-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 4px;
                background: linear-gradient(90deg, #667eea, #764ba2);
            }}
            .metric-card:hover {{
                transform: scale(1.05);
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
            }}
            .metric-value {{
                font-size: 2.5em;
                font-weight: 700;
                color: #667eea;
                display: block;
                margin-bottom: 5px;
            }}
            .metric-label {{
                font-size: 0.9em;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-weight: 500;
            }}
            .metric-change {{
                font-size: 0.8em;
                margin-top: 8px;
                padding: 2px 8px;
                border-radius: 12px;
                display: inline-block;
            }}
            .metric-change.positive {{
                background: #e8f5e8;
                color: #2e7d32;
            }}
            .metric-change.negative {{
                background: #ffebee;
                color: #c62828;
            }}
            .large-chart {{
                grid-column: 1 / -1;
                min-height: 400px;
            }}
            .large-chart .chart-wrapper {{
                height: 400px;
            }}
            .heatmap-container {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 25px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }}
            .heatmap {{
                width: 100%;
                height: 300px;
            }}
            .tooltip {{
                position: absolute;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.3s ease;
                z-index: 1000;
            }}
            @media (max-width: 768px) {{
                .chart-grid {{
                    grid-template-columns: 1fr;
                }}
                .control-group {{
                    flex-direction: column;
                    align-items: stretch;
                }}
                .header h1 {{
                    font-size: 2em;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="header">
                <h1>📊 Interactive Analytics Dashboard</h1>
                <p>Real-time data visualization and business intelligence</p>
            </div>

            <div class="controls">
                <div class="control-group">
                    <label for="timeRange">Time Range:</label>
                    <select id="timeRange" onchange="updateCharts()">
                        <option value="7">Last 7 days</option>
                        <option value="30" selected>Last 30 days</option>
                        <option value="90">Last 90 days</option>
                        <option value="365">Last year</option>
                    </select>

                    <label for="dataSource">Data Source:</label>
                    <select id="dataSource" onchange="updateCharts()">
                        <option value="sales" selected>Sales Data</option>
                        <option value="users">User Analytics</option>
                        <option value="performance">Performance Metrics</option>
                    </select>

                    <button onclick="refreshData()"
                        style="background: #667eea; color: white; border: none; padding: 8px 16px;
                        border-radius: 8px; cursor: pointer;">
                        🔄 Refresh Data
                    </button>
                </div>
            </div>

            <div class="metric-cards" id="metricCards">
                <!-- Metric cards will be populated by JavaScript -->
            </div>

            <div class="chart-grid">
                <div class="chart-container">
                    <div class="chart-title">Revenue Trends</div>
                    <div class="chart-wrapper">
                        <canvas id="revenueChart"></canvas>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">User Engagement</div>
                    <div class="chart-wrapper">
                        <canvas id="engagementChart"></canvas>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">Sales by Category</div>
                    <div class="chart-wrapper">
                        <canvas id="categoryChart"></canvas>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">Performance Distribution</div>
                    <div class="chart-wrapper">
                        <canvas id="performanceChart"></canvas>
                    </div>
                </div>

                <div class="chart-container large-chart">
                    <div class="chart-title">Multi-Metric Analysis</div>
                    <div class="chart-wrapper">
                        <canvas id="multiMetricChart"></canvas>
                    </div>
                </div>
            </div>

            <div class="heatmap-container">
                <div class="chart-title">Activity Heatmap</div>
                <svg class="heatmap" id="heatmap"></svg>
            </div>
        </div>

        <div class="tooltip" id="tooltip"></div>

        <script>
            // Sample data from Python
            const sampleData = {json.dumps(sample_data)};

            let charts = {{}};

            // Initialize dashboard
            document.addEventListener('DOMContentLoaded', function() {{
                initializeMetricCards();
                initializeCharts();
                createHeatmap();
            }});

            function initializeMetricCards() {{
                const container = document.getElementById('metricCards');
                const metrics = [
                    {{
                        value: sampleData.totalRevenue,
                        label: 'Total Revenue',
                        change: '+12.5%',
                        positive: true
                    }},
                    {{
                        value: sampleData.activeUsers.toLocaleString(),
                        label: 'Active Users',
                        change: '+8.3%',
                        positive: true
                    }},
                    {{
                        value: sampleData.conversionRate + '%',
                        label: 'Conversion Rate',
                        change: '-2.1%',
                        positive: false
                    }},
                    {{
                        value: sampleData.averageOrderValue,
                        label: 'Avg Order Value',
                        change: '+15.7%',
                        positive: true
                    }},
                    {{
                        value: sampleData.customerSatisfaction + '/5',
                        label: 'Customer Rating',
                        change: '+0.3',
                        positive: true
                    }},
                    {{
                        value: sampleData.monthlyGrowth + '%',
                        label: 'Monthly Growth',
                        change: '+4.2%',
                        positive: true
                    }}
                ];

                container.innerHTML = metrics.map(metric => `
                    <div class="metric-card">
                        <span class="metric-value">${{metric.value}}</span>
                        <span class="metric-label">${{metric.label}}</span>
                        <div class="metric-change ${{metric.positive ? 'positive' : 'negative'}}">
                            ${{metric.change}}
                        </div>
                    </div>
                `).join('');
            }}

            function initializeCharts() {{
                // Revenue Trends Line Chart
                const revenueCtx = document.getElementById('revenueChart').getContext('2d');
                charts.revenue = new Chart(revenueCtx, {{
                    type: 'line',
                    data: {{
                        labels: sampleData.dates,
                        datasets: [{{
                            label: 'Revenue ($)',
                            data: sampleData.revenueData,
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4,
                            pointBackgroundColor: '#667eea',
                            pointBorderWidth: 0,
                            pointRadius: 4
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                display: false
                            }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                grid: {{
                                    color: 'rgba(0,0,0,0.1)'
                                }}
                            }},
                            x: {{
                                grid: {{
                                    display: false
                                }}
                            }}
                        }}
                    }}
                }});

                // User Engagement Doughnut Chart
                const engagementCtx = document.getElementById('engagementChart').getContext('2d');
                charts.engagement = new Chart(engagementCtx, {{
                    type: 'doughnut',
                    data: {{
                        labels: ['Direct', 'Social Media', 'Search', 'Email', 'Referral'],
                        datasets: [{{
                            data: [30, 25, 20, 15, 10],
                            backgroundColor: [
                                '#667eea',
                                '#764ba2',
                                '#f093fb',
                                '#f5576c',
                                '#4facfe'
                            ],
                            borderWidth: 0
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                position: 'bottom'
                            }}
                        }}
                    }}
                }});

                // Sales by Category Bar Chart
                const categoryCtx = document.getElementById('categoryChart').getContext('2d');
                charts.category = new Chart(categoryCtx, {{
                    type: 'bar',
                    data: {{
                        labels: sampleData.categories,
                        datasets: [{{
                            label: 'Sales',
                            data: sampleData.categoryData,
                            backgroundColor: 'rgba(102, 126, 234, 0.8)',
                            borderColor: '#667eea',
                            borderWidth: 1,
                            borderRadius: 4
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                display: false
                            }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true
                            }}
                        }}
                    }}
                }});

                // Performance Distribution Radar Chart
                const performanceCtx = document.getElementById('performanceChart').getContext('2d');
                charts.performance = new Chart(performanceCtx, {{
                    type: 'radar',
                    data: {{
                        labels: ['Speed', 'Quality', 'Efficiency', 'Innovation', 'Customer Service', 'Reliability'],
                        datasets: [{{
                            label: 'Current',
                            data: [85, 90, 78, 82, 88, 85],
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.2)',
                            borderWidth: 2
                        }}, {{
                            label: 'Target',
                            data: [90, 95, 85, 90, 92, 90],
                            borderColor: '#764ba2',
                            backgroundColor: 'rgba(118, 75, 162, 0.2)',
                            borderWidth: 2
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            r: {{
                                beginAtZero: true,
                                max: 100
                            }}
                        }}
                    }}
                }});

                // Multi-Metric Line Chart
                const multiCtx = document.getElementById('multiMetricChart').getContext('2d');
                charts.multiMetric = new Chart(multiCtx, {{
                    type: 'line',
                    data: {{
                        labels: sampleData.dates,
                        datasets: [
                            {{
                                label: 'Revenue',
                                data: sampleData.revenueData,
                                borderColor: '#667eea',
                                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                                yAxisID: 'y'
                            }},
                            {{
                                label: 'Users',
                                data: sampleData.userData,
                                borderColor: '#764ba2',
                                backgroundColor: 'rgba(118, 75, 162, 0.1)',
                                yAxisID: 'y1'
                            }},
                            {{
                                label: 'Orders',
                                data: sampleData.orderData,
                                borderColor: '#f5576c',
                                backgroundColor: 'rgba(245, 87, 108, 0.1)',
                                yAxisID: 'y2'
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{
                            mode: 'index',
                            intersect: false,
                        }},
                        scales: {{
                            x: {{
                                display: true,
                            }},
                            y: {{
                                type: 'linear',
                                display: true,
                                position: 'left',
                            }},
                            y1: {{
                                type: 'linear',
                                display: true,
                                position: 'right',
                                grid: {{
                                    drawOnChartArea: false,
                                }},
                            }},
                            y2: {{
                                type: 'linear',
                                display: false,
                                position: 'right',
                            }}
                        }}
                    }}
                }});
            }}

            function createHeatmap() {{
                const svg = d3.select('#heatmap');
                const margin = {{top: 20, right: 30, bottom: 40, left: 70}};
                const width = 900 - margin.left - margin.right;
                const height = 300 - margin.bottom - margin.top;

                const data = sampleData.heatmapData;

                const x = d3.scaleBand()
                    .range([0, width])
                    .domain(data.map(d => d.hour))
                    .padding(0.01);

                const y = d3.scaleBand()
                    .range([height, 0])
                    .domain(data.map(d => d.day))
                    .padding(0.01);

                const colorScale = d3.scaleSequential()
                    .interpolator(d3.interpolateBlues)
                    .domain([0, d3.max(data, d => d.value)]);

                const g = svg.append('g')
                    .attr('transform', `translate(${{margin.left}},${{margin.top}})`);

                // Add rectangles
                g.selectAll('.cell')
                    .data(data)
                    .enter().append('rect')
                    .attr('class', 'cell')
                    .attr('x', d => x(d.hour))
                    .attr('y', d => y(d.day))
                    .attr('width', x.bandwidth())
                    .attr('height', y.bandwidth())
                    .style('fill', d => colorScale(d.value))
                    .style('stroke', 'white')
                    .style('stroke-width', 1)
                    .on('mouseover', function(event, d) {{
                        const tooltip = d3.select('#tooltip');
                        tooltip.transition().duration(200).style('opacity', .9);
                        tooltip.html(`Day: ${{d.day}}<br/>Hour: ${{d.hour}}:00<br/>Activity: ${{d.value}}`)
                            .style('left', (event.pageX + 10) + 'px')
                            .style('top', (event.pageY - 28) + 'px');
                    }})
                    .on('mouseout', function() {{
                        d3.select('#tooltip').transition().duration(500).style('opacity', 0);
                    }});

                // Add axes
                g.append('g')
                    .attr('transform', `translate(0,${{height}})`)
                    .call(d3.axisBottom(x));

                g.append('g')
                    .call(d3.axisLeft(y));

                // Add labels
                svg.append('text')
                    .attr('transform', 'rotate(-90)')
                    .attr('y', 0 + 15)
                    .attr('x', 0 - (height / 2))
                    .style('text-anchor', 'middle')
                    .text('Day of Week');

                svg.append('text')
                    .attr('transform', `translate(${{(width / 2) + margin.left}},${{height + margin.top + 35}})`)
                    .style('text-anchor', 'middle')
                    .text('Hour of Day');
            }}

            function updateCharts() {{
                // Simulate data update
                console.log('Updating charts with new parameters');
                // In a real application, you would fetch new data here
            }}

            function refreshData() {{
                // Simulate data refresh with animation
                Object.values(charts).forEach(chart => {{
                    chart.data.datasets.forEach(dataset => {{
                        dataset.data = dataset.data.map(value =>
                            value + (Math.random() - 0.5) * value * 0.2
                        );
                    }});
                    chart.update('active');
                }});

                // Update metric cards
                setTimeout(initializeMetricCards, 500);
            }}
        </script>
    </body>
    </html>
    """

    await flyte.report.replace.aio(html_content)
    await flyte.report.flush.aio()


def generate_sample_data():
    """Generate sample data for the dashboard"""
    # Generate dates for the last 30 days
    dates = []
    base_date = datetime.datetime.now() - datetime.timedelta(days=30)
    for i in range(30):
        date = base_date + datetime.timedelta(days=i)
        dates.append(date.strftime("%m/%d"))

    # Generate sample revenue data with trend
    revenue_data = []
    base_revenue = 10000
    for i in range(30):
        trend = i * 100  # upward trend
        noise = random.randint(-1000, 1000)
        revenue_data.append(max(0, base_revenue + trend + noise))

    # Generate user data
    user_data = []
    base_users = 1000
    for i in range(30):
        trend = i * 10
        noise = random.randint(-100, 100)
        user_data.append(max(0, base_users + trend + noise))

    # Generate order data
    order_data = []
    base_orders = 100
    for i in range(30):
        trend = i * 2
        noise = random.randint(-20, 20)
        order_data.append(max(0, base_orders + trend + noise))

    # Categories and their data
    categories = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports"]
    category_data = [random.randint(500, 2000) for _ in categories]

    # Heatmap data
    heatmap_data = []
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for day in days:
        for hour in range(24):
            # Simulate higher activity during business hours
            if 9 <= hour <= 17:
                base_activity = 50
            elif 18 <= hour <= 22:
                base_activity = 30
            else:
                base_activity = 10

            activity = base_activity + random.randint(0, 20)
            heatmap_data.append({"day": day, "hour": hour, "value": activity})

    return {
        "dates": dates,
        "revenueData": revenue_data,
        "userData": user_data,
        "orderData": order_data,
        "categories": categories,
        "categoryData": category_data,
        "heatmapData": heatmap_data,
        "totalRevenue": f"${sum(revenue_data):,}",
        "activeUsers": sum(user_data),
        "conversionRate": 3.2,
        "averageOrderValue": f"${sum(revenue_data) // sum(order_data):,}",
        "customerSatisfaction": 4.6,
        "monthlyGrowth": 12.5,
    }


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(generate_interactive_dashboard)
    print(run.name)
    print(run.url)
