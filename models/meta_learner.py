import sonnet as snt
import tensorflow as tf
import numpy as np

from models.base_model import BaseModel
import models.layers as Layers
from constants import Constants
from models.embedder import Embedder
from models.encoder import Encoder

META_TRAIN_COMBINED_SUMMARIES = 'meta_train_combined_summaries'

class MetaLearner(BaseModel):
  def __init__(self, name='MetaLearner'):
    super().__init__(name=name)
    self._placeholders = []

  def _strip_name(self, name):
    name_list = name.replace(':', ' ').replace('/', ' ').split()
    if len(name_list) > 2:
      return name_list[1]
    return name_list[0]


  def build_placeholders(self, source_num_way, target_num_way,  grads_weights):
    '''
    Given a list of pairs of (gradient, weight) tensors, builds and returns a list placeholders of the same shape
    '''
    self._source_num_way = source_num_way
    self._target_num_way = target_num_way
    for g, w in grads_weights:
      name = self._strip_name(w.name)
      gradient_ph = tf.placeholder(tf.float32, g.shape,   name=name + '_grad_placeholder')
      weight_ph = tf.placeholder(tf.float32, w.shape, name=name + '_placeholder')
      self._placeholders.append((gradient_ph, weight_ph))
    total = 0
    for grad, weight in self._placeholders:
      total += np.prod(grad.shape.as_list())
      total += np.prod(weight.shape.as_list())
    print("Number of parameters:", total)
    return self._placeholders

  def _build(self, images):
    embedder = Embedder()
    embedded_grads_weights = embedder.embed_all_grads_weights(self._placeholders)
    # Fake batching
    embedded_grads_weights = tf.expand_dims(embedded_grads_weights, 0)
    encoder = Encoder(self._source_num_way, self._target_num_way)
    encoded = encoder.encode(embedded_grads_weights)
    decoded = encoder.decode(encoded)
    # Fake batching
    decoded = tf.squeeze(decoded, [0])
    weight_updates = embedder.unembed_all_weights(decoded)

    the_list = [tf.nn.moments(w, [0]) for w in weight_updates]
    mean_means = tf.reduce_mean([tf.reduce_mean(v[0]) for v in the_list])
    mean_vars = tf.reduce_mean([tf.reduce_mean(v[1]) for v in the_list])
    tf.summary.scalar('weight_updates_mean', mean_means, [META_TRAIN_COMBINED_SUMMARIES])
    tf.summary.scalar('weight_updates_var', mean_vars, [META_TRAIN_COMBINED_SUMMARIES])
    # Get the updated model
    new_weights = [self._placeholders[0][1] + weight_updates[0],
                   self._placeholders[1][1] + weight_updates[1],
                   self._placeholders[2][1] + weight_updates[2],
                   self._placeholders[3][1] + weight_updates[3],
                   self._placeholders[4][1] + weight_updates[4]]
    self.outputs = self.new_model_forward(new_weights, images)
    return self.outputs

  @staticmethod
  def new_model_forward(weights, inputs, make_vars=False):

    # Create tf.Variables if required
    if make_vars:
      weights = [tf.Variable(w) if isinstance(w, np.ndarray) else w for w in weights]

    outputs = tf.nn.conv2d(inputs, weights[0], [1, 1, 1, 1], padding='SAME')
    outputs += weights[1]
    # outputs = tf.nn.bias_add(outputs, )
    outputs = Layers.max_pool(outputs)
    outputs = tf.nn.relu(outputs)

    outputs = tf.nn.conv2d(outputs, weights[2], [1, 1, 1, 1], padding='SAME')
    outputs += weights[3]
    outputs = Layers.max_pool(outputs)
    outputs = tf.nn.relu(outputs)

    outputs = tf.nn.conv2d(outputs, weights[4], [1, 1, 1, 1], padding='SAME')
    outputs += weights[5]
    outputs = Layers.max_pool(outputs)
    outputs = tf.nn.relu(outputs)

    outputs = tf.nn.conv2d(outputs, weights[6], [1, 1, 1, 1], padding='SAME')
    outputs += weights[7]
    outputs = Layers.max_pool(outputs)
    outputs = tf.nn.relu(outputs)

    outputs = tf.nn.conv2d(outputs, weights[8], [1, 1, 1, 1], padding='SAME')
    outputs += weights[9]
    outputs = Layers.global_pool(outputs)
    # Reshape to one-hot predictions
    if isinstance(weights[-1], np.ndarray):
      outputs = tf.reshape(outputs, [-1, weights[-1].shape[-1]])
    else:
      outputs = tf.reshape(outputs, [-1, weights[-1].shape.as_list()[-1]])
    return outputs


  def get_loss(self, labels, outputs):
    '''
    Build and return the loss calculation ops. Assume that graph_nodes contains the nodes you need,
    as a KeyError will be raised if a key is missing.
    '''
    targets = tf.one_hot(tf.to_int32(labels), self._target_num_way)
    targets = tf.Print(targets, [targets[0]], "TARGETS:", summarize=100)
    outputs = tf.Print(outputs, [outputs[0]], "OUTPUTS:", summarize=100)
    return tf.losses.softmax_cross_entropy(targets, outputs)

  def get_target_tensors(self):
    '''
    Returns an arbitrarily nested structure of tensors that are the required input for
    calculating the loss.
    '''
    return tf.placeholder(tf.float32, shape=self.TARGET_SHAPE, name="input_y")

  def training_pass(self, sess, ops, images, labels, grads_weights, summary_op, learning_rate):
    '''
    A single pass through the given batch from the training set
    '''
    feed_dict = {
      ops['images']     : images,
      ops['labels']     : labels,
      ops['learning_rate']: learning_rate,
      ops['is_training']: True
    }
    for (grad, weight), (grad_ph, weight_ph) in zip(grads_weights, self._placeholders):
      feed_dict[grad_ph] = grad
      feed_dict[weight_ph] = weight

    _, loss, outputs, top1, summary = sess.run([
      ops['train_op'],
      ops['loss'],
      ops['outputs'],
      ops['top_1'],
      summary_op
    ], feed_dict)
    return loss, outputs, top1, summary

  def test_pass(self, sess, ops, images, labels, grads_weights, summary_op=None):
    '''
    A single pass through the given batch from the training set
    '''
    feed_dict = {
      ops['images']     : images,
      ops['labels']     : labels,
      ops['is_training']: False
    }
    for (grad, weight), (grad_ph, weight_ph) in zip(grads_weights, self._placeholders):
      feed_dict[grad_ph] = grad
      feed_dict[weight_ph] = weight

    if summary_op is None:
      loss, outputs, top1 = sess.run([
        ops['loss'],
        ops['outputs'],
        ops['top_1'],
      ], feed_dict)
      return loss, outputs, top1
    loss, outputs, top1, summary = sess.run([
      ops['loss'],
      ops['outputs'],
      ops['top_1'],
      summary_op
    ], feed_dict)
    return loss, outputs, top1, summary
