import sys
import os
import random
import csv
import time
from datetime import datetime


import tensorflow as tf
import sonnet as snt
import numpy as np

from constants import Constants
from models.meta_learner import MetaLearner
from data_loader.data_interface import DataInterface

# Define some application flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('write_logs', False, 'If True, writes tensorboard logs. Default: False')

META_TRAIN_COMBINED_SUMMARIES = 'meta_train_combined_summaries'
META_TRAIN_SOURCE_SUMMARIES = 'meta_train_source_summaries'
META_TRAIN_TARGET_SUMMARIES = 'meta_train_target_summaries'
META_TEST_COMBINED_SUMMARIES = 'meta_test_combined_summaries'
META_TEST_SOURCE_SUMMARIES = 'meta_test_source_summaries'
META_TEST_TARGET_SUMMARIES = 'meta_test_target_summaries'


def load_or_init_vars(sess, target_bin_base, init_op):
  '''
  Loads weights from file, or initialises if not saved.
  :param sess: the session to use
  :param target_bin_base: the directory in which `saver/model.meta.ckpt` exists
  :param init_op: the init op to run if no saved weight exist
  '''
  saver_dir = os.path.join(target_bin_base, 'saver')
  if not os.path.exists(saver_dir):
    os.makedirs(saver_dir)
  saver = tf.train.Saver()
  saver_path = os.path.join(saver_dir, 'model.ckpt')
  print(saver_path)
  if os.path.isfile(saver_path + '.meta'):
    print("Loading variables from {}... ".format(saver_path), end='', flush=True)
    saver.restore(sess, saver_path)
  else:
    print("Initializing variables... ", end='', flush=True)
    sess.run(init_op)
  print('complete')
  return saver, saver_path

def build_optimizer():
  if Constants.config['optimizer'] == 'adam':
    return tf.train.AdamOptimizer(learning_rate=Constants.config['learning_rate'])
  elif Constants.config['optimizer'] == 'adadelta':
    return tf.train.AdadeltaOptimizer(learning_rate=Constants.config['learning_rate'])
  elif Constants.config['optimizer'] == 'sgd':
    return tf.train.MomentumOptimizer(learning_rate=Constants.config['learning_rate'], momentum=Constants.config['momentum'])
  else:
    msg = "Optimizer \"" + Constants.config['optimizer'] + "\" not supported."
    raise ValueError(msg)

def get_model_info(subdir_splits, model_index):
  subdir, class_indices = subdir_splits[model_index]
  saver_path = './bin/source_models/{}_{}/{}/saver/model.ckpt.meta'.format(Constants.config['dataset'], Constants.config['source_num_way'], subdir)
  return saver_path, class_indices

def get_new_class_indices(source_indices, is_training):
  num_new_classes = Constants.config['target_num_way'] - Constants.config['source_num_way']
  available_classes = np.arange(0, 80) if is_training else np.arange(80, Constants.config['num_total_classes'])
  to_choose_from = list(set(available_classes) - set(source_indices))
  new_class_indices = np.random.choice(to_choose_from, num_new_classes, replace=False)
  return new_class_indices

def get_grads_weights(data_interface, subdir_splits, model_index, is_training, evaluate=True):
    # Load old model
    g2 = tf.Graph()
    with g2.as_default():
      with tf.Session(graph=g2) as temp_sess:
        source_saver_path, source_indices = get_model_info(subdir_splits, model_index)
        # Load model from file
        print("Loading source model weights from file")

        saver = tf.train.import_meta_graph(source_saver_path)
        saver.restore(temp_sess,tf.train.latest_checkpoint(source_saver_path[:source_saver_path.rindex('/')]))

        # Access all trained weights
        all_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        all_weights_np = [temp_sess.run(w) for w in all_weights]
        # Add uninitialized weights for output layer
        output_weights, output_biases = all_weights_np[-2:]

        new_weights_shape = list(output_weights.shape[:-1]) + [Constants.config['target_num_way'] - Constants.config['source_num_way']]
        new_biases_shape = [Constants.config['target_num_way'] - Constants.config['source_num_way']]



        new_weight_vals = temp_sess.run(tf.truncated_normal(new_weights_shape, mean=np.mean(output_weights), stddev=np.std(output_weights) * 1.25))
        combined_weights = np.concatenate((output_weights, new_weight_vals), -1)
        new_bias_vals = temp_sess.run(tf.truncated_normal(new_biases_shape, mean=np.mean(output_biases), stddev=np.std(output_biases) * 1.25))
        combined_biases = np.concatenate((output_biases, new_bias_vals), -1)
        # temp_sess.run(tf.variables_initializer([combined_weights]))
        all_weights_np[-2:] = combined_weights, combined_biases
        # all_weights_np = [temp_sess.run(w) for w in all_weights]


        # Obtain gradients for target images
        target_indices = get_new_class_indices(source_indices, is_training)
        for t in target_indices:
          assert t not in source_indices, "Overlap in source/target indices. Something is wrong!"
        target_images, target_labels = data_interface.get_next_batch('train', target_indices, num_shot=Constants.config['num_shot'])

        target_labels = np.asarray(target_labels) + Constants.config['source_num_way']
        input_imgs = tf.placeholder(tf.float32, Constants.config['input_shape'])

        predictions = MetaLearner.new_model_forward(all_weights_np, input_imgs, make_vars=True)
        targets = tf.placeholder(tf.float32, [None])

        targets_one_hot = tf.one_hot(tf.to_int32(targets), Constants.config['target_num_way'])
        loss = tf.losses.softmax_cross_entropy(targets_one_hot, predictions)
        temp_sess.run(tf.global_variables_initializer())
        optimizer = tf.train.GradientDescentOptimizer(1e-3)
        grads_weights = optimizer.compute_gradients(loss)
        grads_weights = [(g, w) for (g, w) in grads_weights if g is not None]

        if evaluate:
          print("Computing source model reponse to target images")
          grads_weights = temp_sess.run(grads_weights, {input_imgs: target_images, targets: target_labels})
    return grads_weights, source_indices, target_indices

def build_summary_ops(loss, labels, outputs, ops):
  preds = tf.argmax(outputs, 1)
  preds = tf.Print(preds, [preds], "preds: ", summarize=100)
  labels = tf.Print(labels, [labels], "labels: ", summarize=100)
  equality = tf.equal(tf.to_float(preds), tf.to_float(labels))
  ops['top_1'] = tf.reduce_mean(tf.to_float(equality))
  tf.summary.histogram('predictions', preds, [META_TRAIN_COMBINED_SUMMARIES])
  ops['top1_meta_train_combined'] = tf.summary.scalar('top1_meta_train_combined', ops['top_1'], [META_TRAIN_COMBINED_SUMMARIES])
  ops['top1_meta_train_source'] = tf.summary.scalar('top1_meta_train_source', ops['top_1'], [META_TRAIN_SOURCE_SUMMARIES])
  ops['top1_meta_train_target'] = tf.summary.scalar('top1_meta_train_target', ops['top_1'], [META_TRAIN_TARGET_SUMMARIES])
  ops['top1_meta_test_combined'] = tf.summary.scalar('top1_meta_test_combined', ops['top_1'], [META_TEST_COMBINED_SUMMARIES])
  ops['top1_meta_test_source'] = tf.summary.scalar('top1_meta_test_source', ops['top_1'], [META_TEST_SOURCE_SUMMARIES])
  ops['top1_meta_test_target'] = tf.summary.scalar('top1_meta_test_target', ops['top_1'], [META_TEST_TARGET_SUMMARIES])

  ops['loss_meta_train_combined'] = tf.summary.scalar('loss_meta_train_combined', loss, [META_TRAIN_COMBINED_SUMMARIES])
  ops['loss_meta_train_source'] = tf.summary.scalar('loss_meta_train_source', loss, [META_TRAIN_SOURCE_SUMMARIES])
  ops['loss_meta_train_target'] = tf.summary.scalar('loss_meta_train_target', loss, [META_TRAIN_TARGET_SUMMARIES])
  ops['loss_meta_test_combined'] = tf.summary.scalar('loss_meta_test_combined', loss, [META_TEST_COMBINED_SUMMARIES])
  ops['loss_meta_test_source'] = tf.summary.scalar('loss_meta_test_source', loss, [META_TEST_SOURCE_SUMMARIES])
  ops['loss_meta_test_target'] = tf.summary.scalar('loss_meta_test_target', loss, [META_TEST_TARGET_SUMMARIES])

  ops['meta_train_combined'] = tf.summary.merge_all(META_TRAIN_COMBINED_SUMMARIES)
  ops['meta_train_source'] = tf.summary.merge_all(META_TRAIN_SOURCE_SUMMARIES)
  ops['meta_train_target'] = tf.summary.merge_all(META_TRAIN_TARGET_SUMMARIES)

def training_pass(sess, ops, subdir_splits, model_index, data_interface, meta_learner):
  '''
  A single pass through the given batch from the training set
  '''
  grads_weights, source_indices, target_indices = get_grads_weights(data_interface, subdir_splits, model_index, is_training=True)
  for g, w in grads_weights:
    assert isinstance(g, np.ndarray), "Gradient is not a numpy array"
    assert isinstance(w, np.ndarray), "Weight is not a numpy array"
  print("Performing meta-learner training pass with model {}".format(model_index))
  print("s:", source_indices)
  print("t:", target_indices)
  # Train on TARGET set
  target_images, target_labels = data_interface.get_next_batch('test', target_indices, num_shot=Constants.config['num_shot'])
  # Need to offset the labels by the number of source classes, as we want (for 4 classes -> 6 classes):
  # source classes: [0, 1, 2, 3]; target classes = [4, 5]
  target_labels = np.asarray(target_labels) + Constants.config['source_num_way']
  source_images, source_labels = data_interface.get_next_batch('test', source_indices, num_shot=Constants.config['num_shot'])
  images = np.concatenate((target_images, source_images))
  labels = np.concatenate((target_labels, source_labels))
  summaries = []
  top1s = []
  print("Combined pass")
  loss, outputs, top1, summary = meta_learner.training_pass(sess, ops, images, labels, grads_weights, ops['meta_train_combined'])
  print(top1)
  top1s.append(top1)
  summaries.append(summary)

  print("source pass")
  _, _, top1, summary = meta_learner.test_pass(sess, ops, source_images, source_labels, grads_weights, ops['meta_train_source'])
  print(top1)
  top1s.append(top1)
  summaries.append(summary)

  print("target pass")
  _, _, top1, summary = meta_learner.test_pass(sess, ops, target_images, target_labels, grads_weights, ops['meta_train_target'])
  print(top1)
  top1s.append(top1)
  summaries.append(summary)

  return loss, outputs, top1s, summaries

def test_pass(sess, ops, subdir_splits, model_index, data_interface, meta_learner):
  '''
  A single pass through the given batch from the training set
  '''
  grads_weights, source_indices, target_indices = get_grads_weights(data_interface, subdir_splits, model_index, is_training=False)
  print("Performing meta-learner test pass with model {}".format(model_index))
  print("s:", source_indices)
  print("t:", target_indices)
  # Train on TARGET set
  target_images, target_labels = data_interface.get_next_batch('test', target_indices, num_shot=Constants.config['num_shot'])
  # Need to offset the labels by the number of source classes, as we want (for 4 classes -> 6 classes):
  # source classes: [0, 1, 2, 3]; target classes = [4, 5]
  target_labels = np.asarray(target_labels) + Constants.config['source_num_way']
  source_images, source_labels = data_interface.get_next_batch('test', source_indices, num_shot=Constants.config['num_shot'])
  images = np.concatenate((target_images, source_images))
  labels = np.concatenate((target_labels, source_labels))
  top1s = []
  losses = []
  print("Combined pass")
  loss, outputs, top1 = meta_learner.test_pass(sess, ops, images, labels, grads_weights)
  print(top1)
  top1s.append(top1)
  losses.append(loss)

  print("source pass")
  loss, outputs, top1 = meta_learner.test_pass(sess, ops, source_images, source_labels, grads_weights)
  print(top1)
  top1s.append(top1)
  losses.append(loss)

  print("target pass")
  loss, outputs, top1 = meta_learner.test_pass(sess, ops, target_images, target_labels, grads_weights)
  print(top1)
  top1s.append(top1)
  losses.append(loss)

  return losses, outputs, top1s

def train(config_file, target_bin_base, subdir_splits):
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  data_interface = DataInterface('datasets', Constants.config['dataset'])
  meta_learner = MetaLearner()

  # Build meta-learner placeholders for first-time use.
  grads_weights, _, _ = get_grads_weights(data_interface, subdir_splits, 0, is_training=True, evaluate=False)
  meta_learner.build_placeholders(Constants.config['source_num_way'], Constants.config['target_num_way'], grads_weights)

  ops = {}
  ops['images'] = tf.placeholder(tf.float32,
                                 Constants.config['input_shape'],
                                 name='images')
  ops['global_step'] = tf.train.get_or_create_global_step()
  ops['is_training'] = tf.placeholder(tf.bool,
                                      shape=[],
                                      name='is_training')
  ops['outputs'] = meta_learner(ops['images'])
  ops['labels'] = meta_learner.get_target_tensors()
  ops['loss'] = meta_learner.get_loss(ops['labels'], ops['outputs'])
  print(meta_learner.get_all_variables())
  ops['weight_summary'] = snt.python.modules.util.summarize_variables(meta_learner.get_all_variables())
  opt = build_optimizer()
  grads = opt.compute_gradients(ops['loss'])
  ops['train_op'] = opt.apply_gradients(grads, global_step=ops['global_step'])
  ops['init_op'] = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  build_summary_ops(ops['loss'], ops['labels'], ops['outputs'], ops)
  saver, saver_path = load_or_init_vars(sess, target_bin_base, ops['init_op'])

  if FLAGS.write_logs:
    dir_string = str(datetime.now().strftime('%m-%d_%H:%M_'))
    dir_string += config_file[:config_file.rindex('.')]
    logs_path = os.path.join(target_bin_base, dir_string)
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

  start_time = time.time()

  # source_model_indices = np.arange(71)
  step = sess.run(ops['global_step'])
  last_pass_test = False
  while step < Constants.config['meta_learner_training_steps']:
    loop_start_time = time.time()

    losses = []
    if last_pass_test:
      last_pass_test = False

      source_model_indices = np.arange(71)
      np.random.shuffle(source_model_indices)
      for idx, source_model_index in enumerate(source_model_indices):
        print("TRAIN STEP: {}".format(idx))
        loss, outputs, top1s, local_summaries = training_pass(sess, ops, subdir_splits, source_model_index, data_interface, meta_learner)
        step = sess.run(ops['global_step'])
        if FLAGS.write_logs:
          for summary in local_summaries:
            writer.add_summary(summary, step)
          writer.flush()
        losses.append(loss)
    else:
      last_pass_test = True
      source_model_indices = np.arange(80, len(subdir_splits))
      np.random.shuffle(source_model_indices)
      top1s_combined = []
      top1s_source = []
      top1s_target = []
      losses_combined = []
      losses_source = []
      losses_target = []
      for idx, source_model_index in enumerate(source_model_indices):
        print("TEST STEP: {}".format(idx))
        local_losses, outputs, top1s = test_pass(sess, ops, subdir_splits, source_model_index, data_interface, meta_learner)
        top1s_combined.append(top1s[0])
        top1s_source.append(top1s[1])
        top1s_target.append(top1s[2])
        losses_combined.append(local_losses[0])
        losses_source.append(local_losses[1])
        losses_target.append(local_losses[2])
      if FLAGS.write_logs:
        summary = sess.run(ops['top1_meta_test_combined'], { ops['top_1']: np.mean(top1s_combined) })
        writer.add_summary(summary, step)
        summary = sess.run(ops['top1_meta_test_source'], { ops['top_1']: np.mean(top1s_source) })
        writer.add_summary(summary, step)
        summary = sess.run(ops['top1_meta_test_target'], { ops['top_1']: np.mean(top1s_target) })
        writer.add_summary(summary, step)
        summary = sess.run(ops['loss_meta_test_combined'], { ops['loss']: np.mean(losses_combined) })
        writer.add_summary(summary, step)
        summary = sess.run(ops['loss_meta_test_source'], { ops['loss']: np.mean(losses_source) })
        writer.add_summary(summary, step)
        summary = sess.run(ops['loss_meta_test_target'], { ops['loss']: np.mean(losses_target) })
        writer.add_summary(summary, step)
        writer.flush()

    step = sess.run(ops['global_step'])
    # Display current iteration results
    print("|---Done---+---Step---+--Training Loss--+--Sec/Batch--|")

    time_taken = time.time() - loop_start_time
    percent_done = 100. * step / (Constants.config['meta_learner_training_steps'])
    print("|  {:6.2f}%".format(percent_done) + \
            " | {:8d}".format(int(step)) + \
            " | {:.14s}".format("{:14.6f}".format(np.mean(losses))) + \
            "  | {:.10s}".format("{:10.4f}".format(time_taken)))

    # Save model
    print("=============== SAVING - DO NOT KILL PROCESS UNTIL COMPLETE ==============")
    saver.save(sess, saver_path)
    print("============================== SAVE COMPLETE =============================")
  print("Training complete. Performing test pass")


def main(argv):
  random.seed(7218)
  tf.set_random_seed(6459)
  np.random.seed(7518)

  if FLAGS.write_logs:
    print("Tensorboard data will be written for this run")
  else:
    print("Tensorboard data will NOT be written for this run")
    print("Run application with -h for flag usage")

  config_file = 'conv_cifar10way_proper.yml'
  if '--config_file' in argv:
    config_file = argv[argv.index('--config_file') + 1]
  Constants.load_config(config_file)
  Constants.parse_args(argv)




  source_bin_base = os.path.join('bin', 'source_models', '{}_{}'.format(Constants.config['dataset'], Constants.config['source_num_way']))
  target_bin_base = os.path.join('bin', 'target_models', '{}_{}'.format(Constants.config['dataset'], Constants.config['target_num_way']))

  directories = [(os.path.join(source_bin_base, sub_dir), sub_dir) for sub_dir in os.listdir(source_bin_base) \
                 if os.path.isdir(os.path.join(source_bin_base, sub_dir))]

  # List of (directory, split) pairs
  subdir_splits = []
  with open(os.path.join(source_bin_base, 'idx_splits.csv'), 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
      subdir, *split = row
      subdir_splits.append((subdir, [int(s) for s in split]))

  print("Training meta-learner over {} models".format(len(subdir_splits)))

  train(config_file=config_file, target_bin_base=target_bin_base, subdir_splits=subdir_splits)





if __name__ == "__main__":
  main(sys.argv)
