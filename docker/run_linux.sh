#!/bin/bash
set -e
set -u

if [ $# -eq 0 ]
then
    echo "running docker without display"
    docker run -it --network=host --gpus=all --name=tjur_container tjur
else
    export DISPLAY=$DISPLAY
	echo "setting display to $DISPLAY"
	xhost +
	docker run -it --rm --user=root -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/joshua/Desktop/WORK/tjur:/root/working -e DISPLAY=$DISPLAY --network=host --gpus=all --name=tjur_container tjur
	xhost -
fi
