#!/usr/bin/sh

# create directory
mkdir -p ./data

# download MNIST binary data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P ./data
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P ./data
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ./data
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ./data

# uncompress
gunzip ./data/train-images-idx3-ubyte.gz
gunzip ./data/train-labels-idx1-ubyte.gz
gunzip ./data/t10k-images-idx3-ubyte.gz
gunzip ./data/t10k-labels-idx1-ubyte.gz

