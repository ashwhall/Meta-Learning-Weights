import abc
import cv2
import os

from ..data_augmentation import DataAugmentation

class Label(abc.ABC):
  '''
  Abstract class representing a common interface with labels from different datasets.
  'filename' is a relative filename, 'filepath' is an absolute directory
  '''
  TB_STATISTICS = [] # a list of string names for tensorboard statistics
  def __init__(self, root_dir, video_filename, label_filename, n_frames):
    self.root_dir = root_dir
    self.video_filename = video_filename
    self.label_filename = label_filename
    self.video_filepath = os.path.join(root_dir, video_filename)
    self.label_filepath = os.path.join(root_dir, label_filename)

    self.first_frame = 0
    self.window_size = self._calculate_window_size()
    self.all_targets_values = None

  def inference_targets(self):
    '''
    Returns all targets that are used for comparison at inference time.
    Not necessarily the same as `all_targets`
    '''
    return self.all_targets()

  def all_targets(self):
    '''
    Returns all of targets, and the indices with actual values. Not optimised.
    Has None's from frame 0 to first_frame (non-inclusive)
    Thus the indices of targets align with frame numbers
    '''
    if self.all_targets_values is None:
      self.all_targets_values = []
      for x in range(self.first_frame):
        self.all_targets_values.append(None)
      for x in range(self.first_frame, self.last_frame+1):
        self.all_targets_values.append(self.targets(x))
    return self.all_targets_values, range(self.first_frame, self.last_frame+1)

  @abc.abstractmethod
  def targets(self, frame_num):
    '''
    Looks up and returns the expected targets for frame_num
    '''
    pass

  @abc.abstractmethod
  def evaluate(self, predictions, targets, frame_num):
    '''
    Compares the predictions to the expected results at frame_num. Prints the results
    and returns arbitrary statistics in following format:
    {
      'stat_name': stat_value,
      ...
    }.
    Note that this need not be a direct comparison between targets and predictions.
    '''
    return {}

  @abc.abstractmethod
  def visualise(self, predictions, batch, folder, write_index):
    '''
    Saves file to visualise the comparison between predictions and the target
    '''
    targets = batch.targets
    pass

  @staticmethod
  def average_evaluate(labels, frame_numbers, predictions):
    '''
    Averages the evaluations of a list of labels.
    The labels need not be the same subclass.
    '''
    aggregated = {}
    counts = {}

    # Collect evaluation stats
    for (label, frame_number, prediction) in zip(labels, frame_numbers, predictions):
      evaluate_result = label.evaluate(frame_number, prediction)
      for (name, value) in evaluate_result.items():
        if (name not in aggregated):
          aggregated[name] = 0
          counts[name] = 0
        aggregated[name] += value
        counts[name] += 1

    # Average stats
    return {k: v/counts[k] for (k,v) in aggregated}

  @staticmethod
  def augment_data(frames, targets, augment_options):
    '''
    A thin wrapper for calling functions from DataAugmentation, but by being a
    Label method, allows subclasses to extend the functionality for custom logic
    for certain augment_options
    '''
    for option in augment_options:
      fnc = getattr(DataAugmentation, option['name'])
      frames, targets = fnc(frames, targets, **option['params'])
    return frames, targets
