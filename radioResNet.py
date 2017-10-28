import tensorflow as tf
import numpy as np

def radioNetv2(x,keep_prob):
## defining model ###

    ## layer 1
    W_conv1 = weight_variable([1,5,1,128]) # filter size 1x5
    b_conv1 = bias_variable([256])
    h_conv1 = tf.nn.relu(conv2d(x,W_conv1)+b_conv1)
    #d_conv1 = dropout(h_conv1,keep_prob)

    ## a pooling is required!!

    ## layer 2
    W_conv2 = weight_variable([1,3,64,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_conv1,W_conv2)+b_conv2)

    # layer 3
    W_conv3 = weight_variable([1, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)  ## dense layer

    ## pooling is required!
    #layer 4
    W_conv2 = weight_variable([2, 3, 32, 128])
    b_conv2 = bias_variable([80])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # layer 5
    W_conv2 = weight_variable([2, 3, 32, 128])
    b_conv2 = bias_variable([80])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    ## pooling required!
    ## layer 6
    W_conv2 = weight_variable([1, 3, 256, 80])
    b_conv2 = bias_variable([80])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # fully connected
    ## dense layer 2
    W_fc2 = weight_variable([256,128])
    b_fc2 = bias_variable([128])
    h_fc2 = tf.nn.relu(tf.matmul(d_fc1,W_fc2) + b_fc2)
    d_fc2 = dropout(h_fc2,keep_prob)

    ## Final layer
    W_fc3 = weight_variable([128,10])
    b_fc3 = bias_variable([10])
    y = tf.matmul(d_fc2,W_fc3) + b_fc3

    return y


## This area is dedicated to for the functions Required in the code
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)