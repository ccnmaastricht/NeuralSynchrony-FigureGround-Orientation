FROM mambaorg/micromamba:1.5.6
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml /tmp/environment.yaml
RUN micromamba install --yes --file /tmp/environment.yaml && \
    micromamba clean --all --yes

# Activate the Conda environment for Dockerfile RUN commands
ARG MAMBA_DOCKERFILE_ACTIVATE=1


