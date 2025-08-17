#!/bin/bash

docker run \
    -it \
    --name iago_dedup \
    --runtime nvidia \
    --shm-size=128g \
    --memory=128g \
    -p 8787:8787 \
    -v "$PWD":/workspace \
    nvcr.io/nvidia/nemo:25.07 bash -c "

    apt-get update && apt-get install -y && apt-get install git -y nano && \

    pip install transformers datasets && \
    pip install --extra-index-url https://pypi.nvidia.com nemo-curator[all] && \

    bash
"
