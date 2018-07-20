import tensorflow as tf
import sonnet as snt
import numpy as np
import os
import sys

class Encoder:
  def __init__(self, source_num_way, target_num_way):
    self._source_num_way = source_num_way
    self._target_num_way = target_num_way
  def encode(self, in_tensor):
    in_tensor = snt.Conv2D(output_channels=16, kernel_shape=3, stride=[1, 2], name='encode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2D(output_channels=32, kernel_shape=3, stride=[1, 2], name='encode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2D(output_channels=32, kernel_shape=3, stride=[1, 2], name='encode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2D(output_channels=64, kernel_shape=3, stride=[1, 2], name='encode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2D(output_channels=64, kernel_shape=3, stride=[1, 2], name='encode_conv', padding='SAME')(in_tensor)
    print("Encoded size:", np.prod(in_tensor.shape))
    return in_tensor

  def decode(self, in_tensor):
    in_tensor = snt.Conv2DTranspose(output_channels=64, kernel_shape=3, stride=[1, 2], name='decode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2DTranspose(output_channels=64, kernel_shape=3, stride=[1, 2], name='decode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2DTranspose(output_channels=32, kernel_shape=3, stride=[1, 2], name='decode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2DTranspose(output_channels=32, kernel_shape=3, stride=[1, 2], name='decode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2DTranspose(output_channels=self._target_num_way, kernel_shape=3, stride=[1, 2], name='decode_conv', padding='SAME')(in_tensor)
    in_tensor = snt.Conv2D(output_channels=1, kernel_shape=1, name='decode_conv', padding='SAME')(in_tensor)
    return in_tensor
