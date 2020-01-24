#!/bin/bash

IMAGE=${1:-xdralex/dl-dev:latest}

TENSORBOARD_PORT=6006

WORK=/home/apollo/package

exec docker run --gpus all --shm-size=8192m --rm -u apollo -w $WORK \
  -p $TENSORBOARD_PORT:$TENSORBOARD_PORT \
	-v "$PWD:$WORK" -v "/data/ssd:$WORK/data/ssd" -v "/data/nvme:$WORK/data/nvme" "$IMAGE" \
	python3.7 ./pipeline.py
