import asyncio
import pathlib

import globe_visualization as gv
import interactive_dashboard as id
import protein_3d as p3d
import satellite_images as si
import scatter_3d as s3d
import youtube_embed as yt

import flyte

env = flyte.TaskEnvironment(
    "reports",
    depends_on=[gv.env, id.env, p3d.env, s3d.env, si.env, yt.env],
)


@env.task
async def main():
    tasks = [
        gv.generate_globe_visualization(),
        id.generate_interactive_dashboard(),
        p3d.generate_protein_3d(),
        s3d.generate_scatter_3d(),
        si.generate_satellite_images(),
        yt.generate_youtube_embed(),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml", root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(main)
    print(run.name)
    print(run.url)
