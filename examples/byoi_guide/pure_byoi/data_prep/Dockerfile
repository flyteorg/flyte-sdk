# Team A's image: data preparation
#
# This team owns this entire Dockerfile. They control Python version, WORKDIR,
# and PYTHONPATH. Flyte has no say here.
#
# Pure BYOI constraint: workflow code must be baked in because there is no
# code bundle. Every code change requires rebuilding and pushing this image.

FROM python:3.11-slim

# System deps this team needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Team A's WORKDIR. Python will find modules here because PYTHONPATH includes it.
WORKDIR /app

# Team A's Python packages. These are their own dependencies — Flyte doesn't
# install anything on top in pure BYOI mode.
RUN pip install --no-cache-dir \
    flyte \
    pandas==2.1.4 \
    pyarrow==14.0.1

# Bake the workflow code into the image.
# In pure BYOI, this is the ONLY way Flyte can find your task functions.
# Downside: every edit to tasks.py requires a new image tag + CI build.
COPY workflow_code/ /app/workflow_code/

# /app is on PYTHONPATH so `import workflow_code.tasks` resolves at runtime.
ENV PYTHONPATH=/app
