#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=/home/shahidm/code/caffe4/examples/adaptation/experiments/webcam_to_dslr/protos/solver.prototxt \
    --weights=/home/shahidm/code/caffe4/models/bvlc_alexnet/bvlc_alexnet.caffemodel \
    --gpu 0
