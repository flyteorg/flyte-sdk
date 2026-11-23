ARG EMR_BASE_IMAGE=public.ecr.aws/emr-serverless/spark/emr-7.0.0:latest
FROM ${EMR_BASE_IMAGE}

ARG VERSION

USER root
RUN pip3 install --no-cache-dir --ignore-installed \
    "flyte==${VERSION}" \
    "flyteplugins-emr-serverless==${VERSION}"

USER hadoop:hadoop
