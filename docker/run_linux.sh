#!/bin/bash
set -e
set -u

# 获取脚本所在目录的父目录（绝对路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

if [ $# -eq 0 ]
then
    echo "running docker without display"
    docker run -it --network=host --gpus=all --name=tjur_container tjur
else
    export DISPLAY=$DISPLAY
	echo "setting display to $DISPLAY"
	echo "mounting $PARENT_DIR to /root/working"
	xhost +
	docker run -it --user=root -v /tmp/.X11-unix:/tmp/.X11-unix -v "$PARENT_DIR":/root/working -e DISPLAY=$DISPLAY --network=host --ipc=host --privileged --gpus=all --name=dprm_container dprm
	xhost -
fi
