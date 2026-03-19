from flyteplugins.hitl import event_app_env

import flyte

if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.with_servecontext().serve(event_app_env)
    print(f"App URL: {app.url}")
