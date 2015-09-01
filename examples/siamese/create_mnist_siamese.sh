#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLE="$(readlink -f $(dirname $0))"
TOP_DIR=$EXAMPLE/../..
DATA=$TOP_DIR/data/mnist

echo "Creating leveldb..."

rm -rf $EXAMPLE/mnist_siamese_train_leveldb
rm -rf $EXAMPLE/mnist_siamese_test_leveldb

$EXAMPLE/convert_mnist_siamese_data.bin \
    $DATA/train-images-idx3-ubyte \
    $DATA/train-labels-idx1-ubyte \
    $EXAMPLE/mnist_siamese_train_leveldb
$EXAMPLE/convert_mnist_siamese_data.bin \
    $DATA/t10k-images-idx3-ubyte \
    $DATA/t10k-labels-idx1-ubyte \
    $EXAMPLE/mnist_siamese_test_leveldb

echo "Done."
