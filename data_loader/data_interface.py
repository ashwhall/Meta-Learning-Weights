import importlib
import os

TRAIN = 'train'
TEST = 'test'
EVAL = 'eval'
MODES = [TRAIN, TEST, EVAL]


class DataInterface:
  '''
  The interface for pulling batches of frames. Asynchronously puts batches into
  `batch_queue`
  '''
  def __init__(self, root_dir, dataset_name):
    self._data_loader = DataInterface._load_dataset(root_dir, dataset_name)

  @staticmethod
  def _load_dataset(root_dir, dataset_name):
    '''
    Load the chosen dataset.
    '''
    import_path = root_dir.replace('/', '.') + '.' + dataset_name.lower() + '.loader'

    data_loader_module = importlib.import_module(import_path)
    data_loader_class = getattr(data_loader_module, 'Loader')
    data_loader = data_loader_class(root_dir)
    data_loader.print_dataset_info()

    return data_loader

  def num_classes(self):
    '''
    Returns the number of classes in the dataset
    '''
    return self._data_loader.num_classes()

  def get_next_batch(self, dataset, indices, num_shot):
    '''
    Builds a support/query batch and returns it
    '''
    return self._data_loader.get_next_batch(dataset, indices, num_shot)
