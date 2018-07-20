import tensorflow as tf
import sonnet as snt
import numpy as np


class Embedder:
  def __init__(self):
    # A list of embedding layers for 2d tensors for reusing when the inputs and embeddings are the same size
    # Key is ((in_height, in_width), (emb_height, emb_width))
    self._embedders_2d = {
    }
    self._unembedders_2d = {
    }
    # Convolution padding doesn't work great if we're not using powers of 2 for the embedding sizes
    self._emb_height = 32
    self._emb_width = 32
    self._kernel_height = 3
    self._kernel_width = 3

  def unembed_all_weights(self, weights_list):
    # Separate into a list of tensors (one per gradient/weight pair)
    weights_list = tf.split(weights_list, weights_list.shape.as_list()[0])
    unembedded_grads_weights = []

    for weights, shape in zip(weights_list, self._original_shapes):
      # Split further into individual gradients/weights
      unembedded_grads_weights.append(self._unembed_conv2d_weights(weights, shape))
    return unembedded_grads_weights

  def embed_all_grads_weights(self, grads_weights_list):
    embedded_grads_weights = []
    self._original_shapes = []
    for idx, (grads, weights) in enumerate(grads_weights_list):

      self._original_shapes.append(weights.shape.as_list())
      # if 'conv' in weights.name:
      if idx % 2 == 0:
        embedded_grads_weights.append(self._embed_conv2d_grads_and_weights(grads, weights))
      else:
        embedded_grads_weights.append(self._embed_bias_grads_and_weights(grads, weights))
      # else:
      #   raise ValueError('Layer type not supported - convolutional only at this point')

    return tf.concat(embedded_grads_weights, 0)

  def _embed_2d(self, in_tensor, emb_height=6, emb_width=6, embedding_dict=None):
    '''
    Given a 2d tensor, embeds it into a fixed-size 2d tensor
    '''
    in_height, in_width = in_tensor.shape.as_list()
    lookup_val = ((in_height, in_width), (emb_height, emb_width))
    if lookup_val not in embedding_dict:
      embedding_dict[lookup_val] = {
          'horizontal': snt.Linear(emb_width),
          'vertical': snt.Linear(emb_height)
      }
    embedders = embedding_dict[lookup_val]
    horizontal_embedder = embedders['horizontal']
    vertical_embedder = embedders['vertical']

    compress_horizontal = lambda x: tf.reshape(horizontal_embedder(tf.expand_dims(x, 0)), [emb_width])
    horizontally_compressed = tf.map_fn(compress_horizontal, in_tensor)

    compress_vertical = lambda x: tf.reshape(vertical_embedder(tf.expand_dims(x, 0)), [emb_height])
    vertically_compressed = tf.map_fn(compress_vertical, tf.transpose(horizontally_compressed))
    print('>>>>>>>>>', vertically_compressed.shape)
    return tf.transpose(vertically_compressed)

  def _embed_conv2d(self, in_tensor, emb_height, emb_width, embedders):
    '''
    Embeds a conv2d layer's weights, retaining the spatial size and compressing the rest (assuming 3x3 spatial size)
    '''
    kernel_height, kernel_width, in_channels, out_channels = in_tensor.shape.as_list()
    # print(in_tensor.shape)
    in_tensor = tf.reshape(in_tensor, [kernel_height * kernel_width, in_channels, out_channels])

    embed_fn = lambda x: self._embed_2d(x, emb_height, emb_width, embedders)
    embedded_tensor = tf.map_fn(embed_fn, in_tensor)
    embedded_tensor = tf.reshape(embedded_tensor, [kernel_height, kernel_width, emb_height, emb_width])
    print('Embedded conv: ', embedded_tensor.shape)
    return embedded_tensor

  def _embed_bias_grads_and_weights(self, grads, weights):
    embedded_grads = self._embed_conv2d(grads, emb_height=self._emb_height, emb_width=self._emb_width, embedders=self._embedders_2d)
    embedded_grads_flat = tf.reshape(embedded_grads, [1, -1, 1])
    embedded_weights = self._embed_conv2d(weights, emb_height=self._emb_height, emb_width=self._emb_width, embedders=self._embedders_2d)
    embedded_weights_flat = tf.reshape(embedded_weights, [1, -1, 1])
    print('conv shape: ', tf.concat([embedded_grads_flat, embedded_weights_flat], 2).shape)
    return tf.concat([embedded_grads_flat, embedded_weights_flat], 2)

  def _embed_conv2d_grads_and_weights(self, grads, weights):
    embedded_grads = self._embed_conv2d(grads, emb_height=self._emb_height, emb_width=self._emb_width, embedders=self._embedders_2d)
    embedded_grads_flat = tf.reshape(embedded_grads, [1, -1, 1])
    embedded_weights = self._embed_conv2d(weights, emb_height=self._emb_height, emb_width=self._emb_width, embedders=self._embedders_2d)
    embedded_weights_flat = tf.reshape(embedded_weights, [1, -1, 1])
    print('conv shape: ', tf.concat([embedded_grads_flat, embedded_weights_flat], 2).shape)
    return tf.concat([embedded_grads_flat, embedded_weights_flat], 2)

  def _unembed_conv2d_weights(self, weights, shape):
    weights = tf.reshape(weights, [self._kernel_height, self._kernel_width, self._emb_height, self._emb_width])
    unembedded_weights = self._embed_conv2d(weights, emb_height=shape[-2], emb_width=shape[-1], embedders=self._unembedders_2d)
    return unembedded_weights
