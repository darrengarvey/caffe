#!/usr/bin/env sh

EXAMPLE="$(readlink -f $(dirname $0))"
TOP_DIR=$EXAMPLE/../..
CAFFE="$(find $TOP_DIR -type f -name caffe 2>/dev/null | head -1 || echo caffe)"

cd $TOP_DIR

$CAFFE train --solver=$EXAMPLE/mnist_siamese_solver.prototxt
