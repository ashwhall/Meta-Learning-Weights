class Batch:
  '''
  Essentially the same as BatchItem, but with an extra dimension for a batch
  '''
  def __init__(self, inputs, targets):
    '''
    args:
      inputs (np.array)  - batch of images of shape [batch_size, height, width, depth]
      targets (np.array) - batch of associated target class indices of shape [batch_size]
    '''
    # TODO: Should separate into sample set and query set
    # This will do for now though:
    split_idx = len(inputs)//2
    self.sample_inputs = inputs[:split_idx]
    self.sample_targets = targets[:split_idx]
    self.query_inputs = inputs[split_idx:]
    self.query_targets = targets[split_idx:]
