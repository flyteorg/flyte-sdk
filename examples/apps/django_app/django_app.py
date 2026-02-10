"""A simple Django Hello World app served via AppEnvironment.

This example demonstrates how to create a minimal Django application
and serve it using Flyte's AppEnvironment.

The Django app is structured with separate files:
- settings.py: Django configuration
- urls.py: URL routing and view functions
- wsgi.py: WSGI application for Gunicorn

Deploy
------

Deploy this app using the Flyte CLI:

```
flyte deploy examples/apps/django_app/django_app.py django_app
```

Or run this script directly:

```
python examples/apps/django_app/django_app.py
```
"""

from pathlib import Path

import flyte
import flyte.app


# Create an image with Django and Gunicorn installed
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "django==5.1.4",
    "gunicorn==25.0.3",
)

# Create the AppEnvironment for Django
app_env = flyte.app.AppEnvironment(
    name="django-hello-world",
    image=image,
    args=["gunicorn", "wsgi:application", "--bind", "0.0.0.0:8080"],
    port=8080,
    resources=flyte.Resources(cpu="1", memory="512Mi"),
    requires_auth=False,
    include=["settings.py", "urls.py", "wsgi.py"],
)


if __name__ == "__main__":
    import logging

    flyte.init_from_config(
        root_dir=Path(__file__).parent,
        log_level=logging.DEBUG,
    )
    app = flyte.serve(app_env)
    print(f"Django App URL: {app.url}")
    print(f"API docs: {app.url}/api/info")
