import math
import tensorflow as tf
import sonnet as snt

def max_pool(inputs, strides=2, window=None, padding="SAME"):
  '''
  Helper method for building tf.nn.max_pool ops
  '''
  if window is None:
    window = [1, strides, strides, 1]
  return tf.nn.max_pool(inputs, window, [1, strides, strides, 1], padding)

def avg_pool(inputs, strides=2, padding="SAME"):
  '''
  Helper method for building tf.nn.avg_pool ops
  '''
  return tf.nn.avg_pool(inputs, [1, strides, strides, 1], [1, strides, strides, 1], padding)

def global_pool(inputs):
  '''
  Pool globally over each activation map. Useful for fully-convolutional classifier outputs
  '''
  return tf.nn.max_pool(inputs, [1, inputs.shape[1], inputs.shape[2], 1], [1, 1, 1, 1], padding='VALID')

def conv2d(output_channels, stride=1, rate=1, padding='SAME', kernel_size=3, use_bias=True): # pylint: disable=R0913
  '''
  Helper method for building snt.Conv2D ops
  '''
  return snt.Conv2D(output_channels=output_channels,
                    kernel_shape=kernel_size,
                    stride=stride,
                    padding=padding,
                    rate=rate,
                    use_bias=use_bias)


def random_augment(images):
  '''
  Apply random augmentation to the given images using their default values.
  Assumes that the inputs are normalised [0, 1) images
  '''
  # It's worth noting that contrast should be done first as it can produce unstable results
  # when used after the others - it exaggerates numbers outside of the [0, 1) range
  images = random_contrast(images, clip_outputs=False)
  images = random_brightness(images, clip_outputs=False)
  images = random_colour(images, clip_outputs=False)
  images = random_noise(images, clip_outputs=True)
  return images

def random_brightness(images, intensity=40, clip_outputs=True):
  '''
  Shift the brightness of the given images. Set clip_outputs=False if there are more
  calls to image augmentation functions, so the process isn't repeated.
  Assumes that the inputs are normalised [0, 1) images
  '''
  # As we're using normalised images
  intensity /= 255.
  offset = tf.random_uniform([1], minval=-intensity, maxval=intensity)
  images += offset

  if clip_outputs:
    images = tf.clip_by_value(images, 0, 1)
  return images

def random_colour(images, intensity=20, clip_outputs=True):
  '''
  Shift the colour of the given images. Set clip_outputs=False if there are more
  calls to image augmentation functions, so the process isn't repeated.
  Assumes that the inputs are normalised [0, 1) images
  '''
  # As we're using normalised images
  intensity /= 255.
  offset = tf.random_uniform([3], minval=-intensity, maxval=intensity)
  # This step is required as we may have fused image channels - this way we apply the same
  # alteration to all RGBs
  offset = tf.tile(offset, [tf.shape(images)[-1] // 3])
  images += offset
  if clip_outputs:
    images = tf.clip_by_value(images, 0, 1)
  return images

def random_contrast(images, intensity=0.8, clip_outputs=True):
  '''
  Shift the contrast of the given images. Set clip_outputs=False if there are more
  calls to image augmentation functions, so the process isn't repeated.
  Assumes that the inputs are normalised [0, 1) images
  '''
  factor = tf.random_uniform([1], minval=-intensity, maxval=intensity)
  images = images**((16 + factor)/16)
  if clip_outputs:
    images = tf.clip_by_value(images, 0, 1)
  return images


def random_noise(images, intensity=0.02, clip_outputs=True):
  '''
  Applies random gaussian-noise over the given images. Set clip_outputs=False if there are more
  calls to image augmentation functions, so the process isn't repeated.
  Assumes that the inputs are normalised [0, 1) images
  '''
  noise = tf.random_normal(tf.shape(images), stddev=intensity)
  images += noise
  if clip_outputs:
    images = tf.clip_by_value(images, 0, 1)
  return images
