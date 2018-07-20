import os
import sys
import csv
import random
from multiprocessing import Pool

import tensorflow as tf
import numpy as np

from data_loader.data_interface import DataInterface

# Define some application flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('write_logs', False, 'If True, writes tensorboard logs. Default: False')

class SourceTrainer:
  def __init__(self):
def main(argv):
  np.set_printoptions(precision=4, linewidth=150, suppress=True)
  if FLAGS.write_logs:
    print("Tensorboard data will be written for this run")
  else:
    print("Tensorboard data will NOT be written for this run")
    print("Run application with -h for flag usage")

  if '--config_file' not in argv or '--dataset' not in argv or '--source_num_way' not in argv:
    print("Must provide --config_file, --dataset, --source_num_way")
    print("Example:")
    print("tf train_source_models.py --config_file standard_omniglot.yml --dataset omniglot --source_num_way 50")


  config_file = argv[argv.index('--config_file') + 1]
  dataset = argv[argv.index('--dataset') + 1]
  source_num_way = int(argv[argv.index('--source_num_way') + 1])
  random.seed(7218)
  tf.set_random_seed(6459)
  np.random.seed(7518)

  data_interface = DataInterface('datasets', dataset)

  num_classes = data_interface.num_classes()

  num_parallel = 8

  splits = []
  # Add sequential classes
  for left in np.arange(num_classes - source_num_way + 1):
    right = left + source_num_way
    curr_split = np.arange(left, right)
    splits.append(curr_split)
  indices = np.arange(len(splits))

  print("About to train {} times... that sounds crazy.".format(len(splits)))
  bin_base = os.path.join('bin', 'source_models', "{}_{}".format(dataset, source_num_way))
  if not os.path.exists(bin_base):
    os.makedirs(bin_base)

  with open(os.path.join(bin_base, 'idx_splits.csv'), 'w') as csv_file:
    writer = csv.writer(csv_file)
    for index, split in zip(indices, splits):
      writer.writerow([index, *split])

  def train(index_split):
    index, split = index_split
    bin_dir = os.path.join(bin_base, str(index))
    trainer = SourceTrainer(config_file, description=None, class_indices=split, bin_dir=bin_dir, data_interface=data_interface)
    final_test_score = trainer.run()
    print("Final test score: {:.2f}%".format(100 * final_test_score))
    with open(os.path.join(bin_dir, 'final_score.txt'), 'w') as out_file:
      out_file.write(str(final_test_score))



  pool = Pool(num_parallel)
  pool.map(train, list(zip(indices, splits)))
  print("COMPLETE")

if __name__ == "__main__":
  main(sys.argv)
