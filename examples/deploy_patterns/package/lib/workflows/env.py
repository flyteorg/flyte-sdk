
def get_env_var(key: str, default: str = "") -> str:
    """Get environment variable."""
    import os
    return os.environ.get(key, default)


ENV_CONFIG = {
    "environment": "development",
    "debug": True,
}
