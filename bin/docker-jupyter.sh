#!/bin/bash

IMAGE=${1:-xdralex/dl-dev:latest}

JUPYTER_PORT=4004
TENSORBOARD_PORT=6006

WORK=/home/apollo/package

exec docker run --gpus all --shm-size=8192m --rm -u apollo -w $WORK \
        -p $JUPYTER_PORT:$JUPYTER_PORT -p $TENSORBOARD_PORT:$TENSORBOARD_PORT \
        -v "$PWD:$WORK" -v "/data/ssd:$WORK/data/ssd" -v "/data/nvme:$WORK/data/nvme" "$IMAGE" \
        jupyter-notebook \
        --NotebookApp.notebook_dir=$WORK \
        --NotebookApp.ip=0.0.0.0 --NotebookApp.port=$JUPYTER_PORT \
        --NotebookApp.password_required=False --NotebookApp.token='' \
        --NotebookApp.custom_display_url="http://localhost:$JUPYTER_PORT"
