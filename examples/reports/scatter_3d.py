import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="scatter_3d", image=flyte.Image.from_debian_base().with_pip_packages("plotly", "numpy")
)


@env.task(report=True)
async def generate_scatter_3d():
    import numpy as np
    import plotly.graph_objects as go

    # Create synthetic 3D data over time
    n_points = 50
    n_frames = 30
    np.random.seed(42)

    x = np.random.rand(n_points)
    y = np.random.rand(n_points)
    z = np.random.rand(n_points)

    # Create movement over frames
    frames = []
    for t in range(n_frames):
        dx = 0.1 * np.sin(2 * np.pi * t / n_frames)
        dy = 0.1 * np.cos(2 * np.pi * t / n_frames)
        dz = 0.05 * np.sin(4 * np.pi * t / n_frames)

        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=x + dx,
                    y=y + dy,
                    z=z + dz,
                    mode="markers",
                    marker={
                        "size": 6,
                        "color": z + dz,
                        "colorscale": "Viridis",
                        "opacity": 0.8,
                    },
                    text=[f"Point {i}" for i in range(n_points)],
                    hoverinfo="text",
                )
            ],
            name=f"t{t}",
        )
        frames.append(frame)

    # Create initial figure
    fig = go.Figure(
        data=[frames[0].data[0]],
        layout=go.Layout(
            title="3D Particle Motion Over Time",
            scene={
                "xaxis": {"title": "X"},
                "yaxis": {"title": "Y"},
                "zaxis": {"title": "Z"},
            },
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "method": "animate",
                            "args": [
                                [f.name],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0},
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": f.name,
                        }
                        for f in frames
                    ],
                    "active": 0,
                }
            ],
        ),
        frames=frames,
    )
    await flyte.report.replace.aio(fig.to_html(full_html=True, include_plotlyjs="cdn"))
    await flyte.report.flush.aio()


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    run = flyte.run(generate_scatter_3d)
    print(run.name)
    print(run.url)
