import abc
import sonnet as snt
import tensorflow as tf
import numpy as np

from constants import Constants

class BaseModel(snt.AbstractModule):
  '''
  The abstract base class for all models.
  '''
  TARGET_SHAPE = [None]

  def __init__(self, name):
    super().__init__(name=name)


  @abc.abstractmethod
  def _build(self, inputs, graph_nodes): # pylint: disable=W0221
    '''
    Abstract method - build the Sonnet module.

    Args:
        inputs (tf.Tensor):
        graph_nodes (dict{string->tf.Tensor}): Hooks to common tensors

    Returns:
        outputs (arbitrary structure of tf.Tensors)
    '''
    pass

  @abc.abstractmethod
  def get_loss(self, graph_nodes):
    '''
    Build and return the loss calculation ops. Assume that graph_nodes contains the nodes you need,
    as a KeyError will be raised if a key is missing.
    '''
    pass

  def get_target_tensors(self):
    '''
    Returns an arbitrarily nested structure of tensors that are the required input for
    calculating the loss.
    '''
    return tf.placeholder(tf.float32, shape=self.TARGET_SHAPE, name="input_y")

  def get_predicted_outputs(self, outputs):
    '''
    Given the outputs from _build(), select which parts are predictions based solely on the input video
    '''
    return outputs

  def separate_outputs(self, outputs):
    '''
    Given a batch of outputs from _build(), separate them into an iterable in which
    each element has a single frames' output.
    '''
    return outputs


  @abc.abstractmethod
  def training_pass(self, sess, graph_nodes, train_set):
    '''
    A single pass through the given batch from the training set
    '''
    pass

  @abc.abstractmethod
  def test_pass(self, sess, graph_nodes, test_set):
    '''
    A single pass through the given batch from the training set
    '''
    pass

  def prepare_for_training(self, sess, graph_nodes):
    '''
    Allow the model to do any final preparation before training. This is where the extension of the
    output layer should occur, if necessary
    '''
    pass
