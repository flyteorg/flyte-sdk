FROM python:3.12-slim-bookworm AS connector-slim

LABEL org.opencontainers.image.authors="Flyte Team <users@flyte.org>"
LABEL org.opencontainers.image.source=https://github.com/flyteorg/flyte

ARG VERSION=2.0.0b28

RUN apt-get update && apt-get install build-essential -y \
    && pip install uv

RUN uv pip install --system --no-cache-dir -U --pre \
  flyte==$VERSION \
  flyteplugins-connectors[bigquery]==$VERSION \
  && apt-get clean autoclean \
  && apt-get autoremove --yes \
  && rm -rf /var/lib/{apt,dpkg,cache,log}/ \
  && :

CMD ["flyte", "serve", "connector", "--port", "8000"]
