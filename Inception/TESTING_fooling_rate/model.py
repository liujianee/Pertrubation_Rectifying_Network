
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os.path
import matplotlib.pyplot as plt
import sys, getopt
import zipfile
from timeit import time
from tqdm import tqdm
from tensorflow.contrib.framework import arg_scope

from layers import *
from utils import show_all_variables

class Model(object):
  def __init__(self, config, sess):

    self.config = config
    self.task = config.task
    self.learning_rate = config.learning_rate
    self.max_grad_norm = config.max_grad_norm
    self.input_height = config.input_height
    self.input_width = config.input_width
    self.reg_scale = config.reg_scale
    self.resnet_num = config.resnet_num
    self.bypass_prn = config.bypass_prn
    self.cross_model = config.cross_model

    self.nnclassifier_pretrained_model = config.nnclassifier_pretrained_model

    self._build_placeholders()
    self._build_model(sess)
    self._build_steps()
    #self._build_optim()


  def _build_placeholders(self):
    self.I_x = tf.placeholder(tf.float32, [None, 224, 224, 3], name="I_images")	# I_x: original images
    self.P_x = tf.placeholder(tf.float32, [None, 224, 224, 3], name="P_images")	# P_x: perturbated images
    self.P_y = tf.placeholder(tf.int64, [None], name="P_labels")		# P_y: GT label for P_x 
    self.prn_step = tf.Variable(0, name='prn_step', trainable=False)

    self.P_y_print = tf.Print(self.P_y, [self.P_y], 'GT label(in) = ', summarize=6, first_n=-1)

    print("[*] placeholders created...")

    self.normalized_I_x = normalize(self.I_x)
    self.normalized_P_x = normalize(self.P_x)
    if self.cross_model != 'inception':	# current input is RGB, converted to BGR if rectifier is trained with CaffeNet/VGG
      self.normalized_P_x = tf.reverse(self.normalized_P_x, axis=[-1]) #self.normalized_P_x[:,:,:,[2,1,0]]


  def _build_optim(self):
    def prn_minimize(loss, step, var_list):
      if self.config.optimizer == "sgd":
        optim = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif self.config.optimizer == "adam":
        optim = tf.train.AdamOptimizer(self.learning_rate)
      else:
        raise Exception("[!] Unkown optimizer: {}".format(self.config.optimizer))

      if self.max_grad_norm != None:
        grads_and_vars = optim.compute_gradients(loss)
        new_grads_and_vars = []
        for idx, (grad, var) in enumerate(grads_and_vars):
          if grad is not None and var in var_list:
            new_grads_and_vars.append((tf.clip_by_norm(grad, self.max_grad_norm), var))
        return optim.apply_gradients(new_grads_and_vars,
                                     global_step=step)
      else:
        return optim.minimize(loss, global_step=step, var_list=var_list)

    if self.task == "generative":
      self.prn_optim = prn_minimize(
          self.prn_loss, self.prn_step, self.prn_vars)
    elif self.task == "estimate":
      raise Exception("[!] Not implemented yet")


  def _build_model(self, sess):
    with arg_scope([resnet_block, conv2d, max_pool2d, tanh]):
      self.R_x = self._build_prn(self.normalized_P_x)
      if  self.cross_model != 'inception':	# convert BGR back to RGB for Inception classifier
        self.R_x = tf.reverse(self.R_x, axis=[-1])
      self.denormalized_R_x = denormalize(self.R_x)
      print("[*] PRN created...")
    self.load_nnclassifier(sess)
    self.load_nnbenchmarker(sess)

    self._build_loss()


  def load_nnclassifier(self, sess):
    model = self.nnclassifier_pretrained_model
    
    # Load the Inception model
    with gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        #sess.graph.as_default()

        if self.bypass_prn:
            tf.import_graph_def(graph_def, input_map={'input:0':self.P_x})
            print '[*] PRN is bypassed !' 
        else:
            tf.import_graph_def(graph_def, input_map={'input:0':self.R_x})
            print '[*] PRN is enabled !' 

    self.K_y = sess.graph.get_tensor_by_name("import/softmax2_pre_activation:0")	# predicted label

    self.correct_prediction = tf.equal(tf.argmax(self.K_y,1), self.P_y)
    self.prn_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='prn_accuracy')
    
#    self.K_y = tf.Print(self.K_y, [tf.argmax(self.K_y, 1)], 'argmax(out) = ', summarize=6, first_n=-1)

    print("[*] NN Classifier loaded...")

  def load_nnbenchmarker(self, sess):
    model = self.nnclassifier_pretrained_model
    
    # Load the Inception model
    with gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        #sess.graph.as_default()
        tf.import_graph_def(graph_def, input_map={'input:0':self.I_x})

    self.B_y = sess.graph.get_tensor_by_name("import_1/softmax2_pre_activation:0")	# benchmark label
#    self.B_y = tf.Print(self.B_y, [tf.argmax(self.B_y, 1)], 'argmax(labelling) = ', summarize=6, first_n=-1)

    self.unfooled_prediction = tf.equal(tf.argmax(self.K_y,1), tf.argmax(self.B_y,1))
    self.benchmark_prediction = tf.equal(tf.argmax(self.B_y,1), self.P_y)

    self.unfooling_rate = tf.reduce_mean(tf.cast(self.unfooled_prediction, tf.float32), name='unfooling_rate')
    self.fooling_rate = 1.0 - self.unfooling_rate

    self.benchmark_accuracy = tf.reduce_mean(tf.cast(self.benchmark_prediction, tf.float32), name='benchmark_accuracy')

    print("[*] NN Benchmarker loaded...")


  def _build_prn(self, layer):
    with tf.variable_scope("PRN") as sc:
      layer = conv2d(layer, 64, 3, 1, activation_fn=None, scope="conv_0")
      layer = repeat(layer, self.resnet_num, resnet_block, scope="resnet")

      #layer = conv2d(layer, 3, 1, 1, activation_fn=None, scope="conv_1")
      #output = tf.nn.relu(layer, name="relu_output")
      #output = tanh(layer, name="tanh_output")

      layer = conv2d(layer, 16, 1, 1, activation_fn=None, scope="conv_11")

      output = conv2d(layer, 3, 1, 1, activation_fn=None, scope="conv_1")

      self.prn_vars = tf.contrib.framework.get_variables(sc)
    return output


  def _build_loss(self):
    # prn loss
    def log_loss(logits, label, name):
      return tf.reduce_mean(SE_loss(logits=logits, labels=label), name=name)

    with tf.name_scope("build_loss"):
      #self.pert_loss = log_loss(self.K_y, self.P_y_print, "pert_loss")
      self.pert_loss = log_loss(self.K_y, tf.argmax(self.B_y,1), "pert_loss")
      self.reg_loss = self.reg_scale * tf.reduce_sum(
              #tf.abs(self.R_x - self.normalized_P_x), [1, 2, 3],
              tf.abs(self.R_x - self.normalized_I_x), [1, 2, 3],
              name="regularization_loss")
      #self.prn_loss = tf.reduce_mean(self.pert_loss + self.reg_loss, name="prn_loss")
      self.prn_loss = tf.reduce_mean(self.pert_loss)

    self.prn_summary = tf.summary.merge([
        tf.summary.scalar("build_loss/pert_loss",
                          tf.reduce_mean(self.pert_loss)),
        tf.summary.scalar("build_loss/regularization_loss",
                          tf.reduce_mean(self.reg_loss)),
        tf.summary.scalar("build_loss/prn_loss",
                          tf.reduce_mean(self.prn_loss)),
        tf.summary.scalar("prn_accuracy", self.prn_accuracy),
    ])

    self.test_acc_summary = tf.summary.scalar("test_accuracy", self.prn_accuracy)

    print("[*] loss built...")


  def _build_steps(self):
    def prn_run(sess, feed_dict, fetch,
                summary_op, summary_writer, output_op=None):
      if summary_writer is not None:
        fetch['summary'] = summary_op
      if output_op is not None:
        fetch['output'] = output_op

      result = sess.run(fetch, feed_dict=feed_dict)
      if result.has_key('summary'):
        summary_writer.add_summary(result['summary'], result['step'])
        summary_writer.flush()
      return result

    def train_prn(sess, feed_dict, summary_writer=None, with_output=False):
      fetch = {
          'loss': self.prn_loss,
          'optim': self.prn_optim,
          'step': self.prn_step,
      }
      return prn_run(sess, feed_dict, fetch,
                     self.prn_summary, summary_writer,
                     output_op=self.R_x if with_output else None)

    def test_prn(sess, feed_dict, summary_writer=None, with_output=False):
      fetch = {
          #'test_loss': self.prn_loss,
          'fooling_rate': self.fooling_rate,
          'benchmark_acc': self.benchmark_accuracy,
          'prn_acc': self.prn_accuracy,
          'step': self.prn_step,
      }
      
      if summary_writer is not None:
          fetch['summary'] = self.test_acc_summary
      
      result = sess.run(fetch, feed_dict=feed_dict)
      if result.has_key('summary'):
          summary_writer.add_summary(result['summary'], result['step'])
          summary_writer.flush()
      return result

    self.train_prn = train_prn
    self.test_prn = test_prn
