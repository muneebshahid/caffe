#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=/home/shahidm/code/caffe4/examples/adaptation/experiments/amazon_to_webcam/protos/solver.prototxt \
    --weights=/home/shahidm/code/caffe4/models/bvlc_alexnet/bvlc_alexnet.caffemodel \
    --gpu 0
