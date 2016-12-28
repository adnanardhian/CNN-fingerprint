# cnn-fingerprint
fingerprint verification using convolutional neural networks

In this project, the architecture of the cnn is same as mnist handwritten digits architecture, thanks to
https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

also using keras library, with Theano as backend

The fingerprint dataset is from NIST dataset, thanks to https://www.nist.gov
Total data used is 2400 fingerprints, which paired into 1200 paired fingerprints, 600 are identical, 600 are not identical
from total 1200, 1000 is used for training, and 200 is used for testing

The average accuracy from 10 times experiments with same data and configuration is 73,90%
