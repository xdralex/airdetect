#!/bin/bash

IMAGE=xdralex/dl-dev:latest
WORK=/home/apollo/package

if [[ "$1" == "--tensorboard" ]]
then
  CONT_TENSORBOARD_PORT=6006
  HOST_TENSORBOARD_PORT=$2
  PORT_BINDING="-p $HOST_TENSORBOARD_PORT:$CONT_TENSORBOARD_PORT"
  shift; shift
else
  PORT_BINDING=""
fi

exec docker run --gpus all --shm-size=8192m -i -t --rm -u apollo -w $WORK $PORT_BINDING \
	-v "$PWD:$WORK" -v "/data/ssd:$WORK/data/ssd" -v "/data/nvme:$WORK/data/nvme" "$IMAGE" \
	python3.7 pipeline/pipeline.py "${@:1}"
