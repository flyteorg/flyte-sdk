# Team A's image: data preparation
# Uses conda. Team A has no knowledge of Flyte.
# Python is at /opt/conda/bin/python.
# No PYTHONPATH set — teams don't know Flyte needs it.

FROM continuumio/miniconda3:latest

RUN conda install -y -c conda-forge pandas==2.2.3 pyarrow==19.0.1 && \
    conda clean -afy

WORKDIR /app
