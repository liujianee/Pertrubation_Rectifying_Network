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
    self.input_height = config.input_height
    self.input_width = config.input_width
    self.batch_size = config.batch_size
    self.log_step = config.log_step
    self.net_type = config.net_type
    self.perturb_dir = config.perturb_dir
    self.pert_train_list = config.pert_train_list
    #self.pert_test_list = config.pert_test_list
    self.pert_test_dir = config.pert_test_dir
    self.bypass_prn = config.bypass_prn
    self.perturb_en = config.perturb_en
    self.perturb_mode = config.perturb_mode
    self.eval_protocol = config.eval_protocol
    self.testcase_ID = config.testcase_ID
    self.test_log_file = 'loglog_testcase_' + self.testcase_ID + '.log'

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

    # define variables to restore here
    exclude = ['prn_step']
    variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
    self.saver = tf.train.Saver(variables_to_restore, max_to_keep=10)

    self.summary_writer = tf.summary.FileWriter(self.tblog_dir)

  def train(self):
    print("[*] Training starts...")
    self._summary_writer = None

    train_set = prn_data.get_dataset(self.data_dir)
    
    # Get a list of image paths and their labels
    #image_list, label_list = prn_data.get_image_paths_and_labels(train_set)
    if self.eval_protocol == 'A':
      image_list, label_list = prn_data.get_image_paths_and_labels_TEST(train_set)
    elif self.eval_protocol == 'B':
      image_list, label_list = prn_data.get_image_paths_and_labels_TEST_2(train_set)
    assert len(image_list)>0, 'The dataset should not be empty'

    print('Total number of classes: %d' % len(train_set))
    print('Total number of examples: %d' % len(image_list))

    # convert string into tensors
    train_images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    train_labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

    # create input queues
    train_input_queue = tf.train.slice_input_producer(
                                        [train_images, train_labels],
                                        shuffle=True)

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
      #print curr_lab_batch
      file_perturbation = self.perturb_dir + random.choice(self.pert_train_list)
      v = np.load(file_perturbation)

      curr_img_batch_o = preprocess_image_batch(curr_img_batch)
      curr_img_batch_p = np.zeros_like(curr_img_batch_o)

      for ix, one_img in enumerate(curr_img_batch_o):
        clipped_v = np.clip(undo_image_avg(curr_img_batch_o[ix,:,:,:]+v[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(curr_img_batch_p[ix,:,:,:]), 0, 255)
        curr_img_batch_p[ix,:,:,:] = curr_img_batch_o[ix,:,:,:] + clipped_v[None, :, :, :]

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

    def test_prn():
      curr_img_batch, curr_lab_batch = self.sess.run([train_image_batch, train_label_batch])

      self.pert_test_list = glob.glob(self.perturb_dir + self.pert_test_dir + '*.npy')
      file_perturbation = random.choice(self.pert_test_list)	# Testing
      v = np.load(file_perturbation)

      curr_img_batch_o = preprocess_image_batch(curr_img_batch)
      curr_img_batch_p = np.zeros_like(curr_img_batch_o)

      if self.perturb_en:
        if self.perturb_mode == 0:
          print '[*] Inputing pertrubed images'
          for ix, one_img in enumerate(curr_img_batch_o):
            clipped_v = np.clip(undo_image_avg(curr_img_batch_o[ix,:,:,:]+v[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(curr_img_batch_o[ix,:,:,:]), 0, 255)
            curr_img_batch_p[ix,:,:,:] = curr_img_batch_o[ix,:,:,:] + clipped_v[None, :, :, :]

        elif self.perturb_mode == 1:
          print '[*] Inputing pertrubed & clean images'
          for ix, one_img in enumerate(curr_img_batch_o):
            if bool(random.getrandbits(1)):
              clipped_v = np.clip(undo_image_avg(curr_img_batch_o[ix,:,:,:]+v[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(curr_img_batch_o[ix,:,:,:]), 0, 255)
              curr_img_batch_p[ix,:,:,:] = curr_img_batch_o[ix,:,:,:] + clipped_v[None, :, :, :]
            else:
              curr_img_batch_p[ix,:,:,:] = curr_img_batch_o[ix,:,:,:]
      else:
        print '[*] Inputing clean images'
        curr_img_batch_p = curr_img_batch_o


      feed_dict = {
        self.model.I_x: curr_img_batch_o,
        self.model.P_x: curr_img_batch_p,
        self.model.P_y: curr_lab_batch
      }

      res = self.model.test_prn(self.sess, feed_dict, self._summary_writer, with_output=True)
      self._summary_writer = self._get_summary_writer(res)
      #return res['test_acc']
      return res

    fooling_rate_list = []
    benchmark_acc_list = []
    prn_acc_list = []
    for step in trange(self.max_step, desc="Test prn"):
      results = test_prn()
      print '[*] folling rate is %f' % results['fooling_rate']
      print '[*] benchmark acc is %f' % results['benchmark_acc']
      print '[*] prn accuracy is %f' % results['prn_acc']

      fooling_rate_list.append(results['fooling_rate'])
      benchmark_acc_list.append(results['benchmark_acc'])
      prn_acc_list.append(results['prn_acc'])

    fooling_rate_arr = np.array(fooling_rate_list)
    benchmark_acc_arr = np.array(benchmark_acc_list)
    prn_acc_arr = np.array(prn_acc_list)

    print '[**] Average Fooling Rate is %f' % fooling_rate_arr.mean()
    print '[**] Average Benchmark Accuracy is %f' % benchmark_acc_arr.mean()
    print '[**] Average PRN Accuracy is %f' % prn_acc_arr.mean()


    with open(self.test_log_file, 'w') as log_write_f:
      log_write_f.write('Testcase_ID:{}\nbypass_prn:{}, perturb_en:{}, perturb_mode:{}\nData_dir:{}\nPert_dir:{}\nPRN_model:{}\n \
                         \n \
                         [**] Average Fooling Rate is \t{}\n \
                         [**] Average Benchmark Acc is \t{}\n \
                         [**] Average PRN Accuracy is \t{}\n \
                         '.format(self.testcase_ID, str(self.bypass_prn), str(self.perturb_en), str(self.perturb_mode), \
                         self.data_dir, self.pert_test_dir, self.pretrained_model, \
                         str(fooling_rate_arr.mean()), \
                         str(benchmark_acc_arr.mean()), \
                         str(prn_acc_arr.mean()) \
                       ))

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
