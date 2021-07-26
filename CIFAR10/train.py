"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
import csv
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from keras.utils import np_utils

from model import Model
import cifar10_input
from pgd_attack import LinfPGDAttack

from basics import *
from searchMCTS import searchMCTS

import cifar_nn as NN
from model_pre import model_pre
from sift_mcts import *


with open('config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train')

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
    total_loss,
    global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_input)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

with tf.Session() as sess:

  # initialize data augmentation
  
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Initialize statistic excel table
  tuples = [['nat_acc', 'adv_acc', 'nat_xent', 'adv_xent']]
  row = [0,0,0,0]
  tuples.append(row)
  print(tuples)
  with open('stat.csv', 'w', newline ='') as fout:
    csvout = csv.writer(fout)
    csvout.writerows(tuples)  

  # Initialize mcts adversarial variables
  train_size = 1000
  train_ad_images = np.zeros((train_size, 32, 32, 3), np.float32)
  train_ad_labels = np.zeros(train_size, dtype='int32')

  # Setting up mcts adversarial data
  mcts_ad_im = config['mcts_ad_images']
  mcts_ad_lb = config['mcts_ad_lables']
  if os.path.exists(mcts_ad_im):
    train_ad_images = np.load(mcts_ad_im)
    train_ad_labels = np.load(mcts_ad_lb)
  else:
    # Pre-training model
    model_pre = model_pre()
    
    # Generate MCTS Adversarial Perturbation Sets
    for bb in range(train_size):
      pre_image = cifar.train_data.raw_datasubset.xs[bb]
      pre_image = pre_image.reshape(32, 32, 3)
      pre_image = pre_image / 255
      train_ad_images[bb], found, originalClass, newClass, newConfident, eudist, l1dist, l0dist, percent = sift_mcts_single(model_pre, pre_image)
      train_ad_labels[bb] = cifar.train_data.raw_datasubset.ys[bb]
      print('AD IMAGE:', bb)


    np.save('train_ad_mcts', train_ad_images)
    np.save('train_lb_mcts', train_ad_labels)

  
  # Main training loop
  
  total_adnum = 0
  total_eudist = 0.0
  total_l1dist = 0.0
  total_l0dist = 0.0
  total_percent = 0.0
  total_natcross_ent = 0.0
  total_advcross_ent = 0.0

  # Main training loop
  for ii in range(max_num_training_steps):
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                       multiple_passes=True)

    
    # Generate random index of adversarial examples for each batch
    random_index = np.zeros(batch_size, dtype='int32')

    for xx in range(batch_size):
  	  random_index[xx] = random.randint(0, train_size-1)
    print('random index(0-%s): %s '%(train_size, random_index))
    
    ad_image = np.zeros((batch_size, 32, 32, 3), dtype=np.float32)
    for kk in range(batch_size):
      ad_image[kk] = train_ad_images[random_index[kk]]
      x_batch[kk] = cifar.train_data.raw_datasubset.xs[random_index[kk]]
      y_batch[kk] = train_ad_labels[random_index[kk]]

    print('y_batch(0-9): ', y_batch)

    
    # Compute Adversarial Perturbations
    start = timer()
    #x_batch_adv = attack.perturb(x_batch, y_batch, sess)

    ad_image1 = (ad_image*255).astype(np.uint8)
    x_batch_adv = ad_image1.reshape(batch_size, 32, 32, 3)

    end = timer()
    training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    adv_dict = {model.x_input: x_batch_adv,
                model.y_input: y_batch}
    
    total_natcross_ent = total_natcross_ent + sess.run(model.xent, feed_dict=nat_dict)
    total_advcross_ent = total_advcross_ent + sess.run(model.xent, feed_dict=adv_dict)

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      nat_xent = sess.run(model.xent, feed_dict=nat_dict)
      adv_xent = sess.run(model.xent, feed_dict=adv_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      print('    training nat xent     {:.4}'.format(nat_xent))
      print('    training adv xent     {:.4}'.format(adv_xent))
      print('    Avg training nat xent     {:.4}'.format(total_natcross_ent/ii))
      print('    Avg training adv xent     {:.4}'.format(total_advcross_ent/ii))
      #print('    ave eudist: %.4s  avg l1dist: %.4s avg l0dist: %.4s avg per: %s'%(total_eudist/float(total_adnum), total_l1dist/float(total_adnum), total_l0dist/float(total_adnum), total_percent/float(total_adnum)))
      
      row = [nat_acc,adv_acc,nat_xent,adv_xent]
      with open('stat.csv', 'a', newline ='') as fout:
        csvout = csv.writer(fout)
        csvout.writerows([row])


      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=adv_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=adv_dict)
    end = timer()
    training_time += end - start
