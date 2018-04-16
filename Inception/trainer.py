import os
import numpy as np
from tqdm import trange
import random
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
import matplotlib.pyplot as plt
import glob

import prn_data
from prn_data import preprocess_image_batch, undo_image_avg, save_variables_and_metagraph
from model import Model

class Trainer(object):
  def __init__(self, config):
    self.sess = tf.Session()

    self.config = config
    self.max_step = config.max_step
    self.model_dir = config.model_dir
    self.tblog_dir = config.tblog_dir
    self.gpu_memory_fraction = config.gpu_memory_fraction
    self.data_dir = config.data_dir
    self.data_dir_1 = config.data_dir_1
    self.data_dir_2 = config.data_dir_2
    self.data_dir_3 = config.data_dir_3
    self.data_dir_4 = config.data_dir_4
    self.input_height = config.input_height
    self.input_width = config.input_width
    self.batch_size = config.batch_size
    self.log_step = config.log_step
    self.net_type = config.net_type
    self.perturb_dir = config.perturb_dir
    self.pert_train_dir = config.pert_train_dir
    #self.pert_test_list = config.pert_test_list
    self.pretrained_model = config.pretrained_model

    self.model = Model(config, self.sess)

    self.summary_ops = {
        'original_images': {
            'summary': tf.summary.image("original_images",
                                        self.model.I_x,
                                        max_outputs=config.max_image_summary),
            'output': self.model.I_x,
        },
        'pre_prn_images': {
            'summary': tf.summary.image("pre_prn_images",
                                        self.model.P_x,
                                        max_outputs=config.max_image_summary),
            'output': self.model.P_x,
        },
        'prn_images': {
            'summary': tf.summary.image("prn_images",
                                        self.model.denormalized_R_x,
                                        max_outputs=config.max_image_summary),
            'output': self.model.denormalized_R_x,
        }
    }

    self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
    self.summary_writer = tf.summary.FileWriter(self.tblog_dir)

  def train(self):
    print("[*] Training starts...")
    self._summary_writer = None

    train_set_0 = prn_data.get_dataset(self.data_dir)
    train_set_1 = prn_data.get_dataset(self.data_dir_1)
    train_set_2 = prn_data.get_dataset(self.data_dir_2)
    train_set_3 = prn_data.get_dataset(self.data_dir_3)
    train_set_4 = prn_data.get_dataset(self.data_dir_4)
    
    # Get a list of image paths and their labels
    #image_list, label_list = prn_data.get_image_paths_and_labels(train_set)
    image_list_0, label_list_0 = prn_data.get_image_paths_and_labels_TRAIN(train_set_0)
    image_list_1, label_list_1 = prn_data.get_image_paths_and_labels_TRAIN(train_set_1)
    image_list_2, label_list_2 = prn_data.get_image_paths_and_labels_TRAIN(train_set_2)
    image_list_3, label_list_3 = prn_data.get_image_paths_and_labels_TRAIN(train_set_3)
    image_list_4, label_list_4 = prn_data.get_image_paths_and_labels_TRAIN(train_set_4)

    image_list = image_list_0 + image_list_1 + image_list_2 + image_list_3 + image_list_4
    label_list = label_list_0 + label_list_1 + label_list_2 + label_list_3 + label_list_4

    assert len(image_list)>0, 'The dataset should not be empty'

    #print('Total number of classes: %d' % len(train_set))
    print('Total number of examples: %d' % len(image_list))

    # convert string into tensors
    train_images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    train_labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

    # create input queues
    train_input_queue = tf.train.slice_input_producer(
                                        [train_images, train_labels],
                                        shuffle=True)
                                        #shuffle=False)

    # process path and string tensor into an image and a label
    file_content = tf.read_file(train_input_queue[0])
    #train_image = tf.image.decode_jpeg(file_content, channels=3)
    train_image = tf.image.decode_png(file_content, channels=3)
    
    #train_image = tf.image.resize_images(train_image, [self.input_height, self.input_width])
    
    train_label = train_input_queue[1]

    train_image.set_shape([self.input_height, self.input_width, 3])

    # collect batches of images before processing
    train_image_batch, train_label_batch = tf.train.batch(
                                        [train_image, train_label],
                                        batch_size=self.batch_size
                                        )


    print "=input pipeline ready="

    self.sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)   

    if self.pretrained_model:
        print '[*] Loading pretrained model..'
        self.saver.restore(self.sess, self.pretrained_model)


    def train_prn():
      curr_img_batch, curr_lab_batch = self.sess.run([train_image_batch, train_label_batch])

      self.pert_train_list = glob.glob(self.perturb_dir + self.pert_train_dir + '*.npy')
      #file_perturbation = self.perturb_dir + random.choice(self.pert_train_list)
      #v = np.load(file_perturbation)

      curr_img_batch_o = preprocess_image_batch(curr_img_batch)	# original images
      curr_img_batch_p = np.zeros_like(curr_img_batch_o)

      for ix, one_img in enumerate(curr_img_batch_o):
        if bool(random.getrandbits(1)):	# randomly perturb the original image and then feed into the network
        #if True:	# always perturb input images
          file_perturbation = random.choice(self.pert_train_list)
          v = np.load(file_perturbation)
          clipped_v = np.clip(undo_image_avg(curr_img_batch_o[ix,:,:,:]+v[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(curr_img_batch_o[ix,:,:,:]), 0, 255)
          curr_img_batch_p[ix,:,:,:] = curr_img_batch_o[ix,:,:,:] + clipped_v[None, :, :, :]
        else:
          curr_img_batch_p[ix,:,:,:] = curr_img_batch_o[ix,:,:,:]

      feed_dict = {
        self.model.I_x: curr_img_batch_o,
        self.model.P_x: curr_img_batch_p,
        self.model.P_y: curr_lab_batch
      }

      res = self.model.train_prn(self.sess, feed_dict, self._summary_writer, with_output=True)
      self._summary_writer = self._get_summary_writer(res)

      if res['step'] % self.log_step == 0:
        #feed_dict = {
        #  self.model.P_x: curr_img_batch,
        #  self.model.P_y: curr_lab_batch
        #}
        self._inject_summary(
          'original_images', feed_dict, res['step'])
        self._inject_summary(
          'pre_prn_images', feed_dict, res['step'])
        self._inject_summary(
          'prn_images', feed_dict, res['step'])

        print ('training acc is now: %f' % self.sess.run(self.model.prn_accuracy, feed_dict=feed_dict))

    # def test_prn():
    #   curr_img_batch, curr_lab_batch = self.sess.run([train_image_batch, train_label_batch])
    #   #print curr_lab_batch
    #   file_perturbation = self.perturb_dir + random.choice(self.pert_test_list)	# Testing
    #   v = np.load(file_perturbation)

    #   curr_img_batch_p = preprocess_image_batch(curr_img_batch)

    #   for ix, one_img in enumerate(curr_img_batch_p):
    #       clipped_v = np.clip(undo_image_avg(curr_img_batch_p[ix,:,:,:]+v[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(curr_img_batch_p[ix,:,:,:]), 0, 255)
    #       curr_img_batch_p[ix,:,:,:] = curr_img_batch_p[ix,:,:,:] + clipped_v[None, :, :, :]

    #   feed_dict = {
    #     self.model.P_x: curr_img_batch_p,
    #     self.model.P_y: curr_lab_batch
    #   }

    #   res = self.model.test_prn(self.sess, feed_dict, self._summary_writer, with_output=True)
    #   self._summary_writer = self._get_summary_writer(res)

    for step in trange(self.max_step, desc="Train prn"):
      train_prn()
      #if step %  (self.max_step/100) == 0:
      if step %  200 == 0:
          save_variables_and_metagraph(self.sess, self.saver, self.model_dir, self.net_type, step)
      #test_prn()

    #writer = tf.summary.FileWriter('dbg_logs', self.sess.graph)
    #writer.close()


  def _inject_summary(self, tag, feed_dict, step):
    summaries = self.sess.run(self.summary_ops[tag], feed_dict)
    self.summary_writer.add_summary(summaries['summary'], step)


  def _get_summary_writer(self, result):
    if result['step'] % self.log_step == 0:
      return self.summary_writer
    else:
      return None
