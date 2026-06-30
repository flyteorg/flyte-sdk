from my_app.env import env


@env.task
async def compute_stats(values: list[float]) -> dict:
    """Compute basic statistics using the my_lib utility library."""
    from my_lib.stats import mean, std

    return {
        "mean": mean(values),
        "std": std(values),
        "count": len(values),
    }


@env.task
async def summarize(stats: dict) -> str:
    return f"n={stats['count']}, mean={stats['mean']:.2f}, std={stats['std']:.2f}"
