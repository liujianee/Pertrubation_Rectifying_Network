
import os
import time
#from subprocess import Popen, PIPE
#import tensorflow as tf
#from tensorflow.python.framework import ops
import numpy as np
#from scipy import misc
#from sklearn.model_selection import KFold
#from scipy import interpolate
#from tensorflow.python.training import training
import random
import re
from scipy.misc import imread, imresize
import scipy.io as sio
#from tensorflow.python.platform import gfile


class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
def sort_nicely(l):
    # Sort the given list in the way that humans expect.
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )


def get_dataset(paths, has_class_directories=True):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        #classes.sort()
        sort_nicely(classes)
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            image_paths = get_image_paths(facedir)
            dataset.append(ImageClass(class_name, image_paths))
    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i+1] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def get_image_paths_and_labels_TRAIN(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_ = dataset[i].image_paths
        sort_nicely(image_paths_)
        image_paths_flat += image_paths_[0:40]
        labels_flat += [i+1] * len(image_paths_[0:40])
    return image_paths_flat, labels_flat

def get_image_paths_and_labels_TEST(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_ = dataset[i].image_paths
        sort_nicely(image_paths_)
        image_paths_flat += image_paths_[40:50]
        labels_flat += [i+1] * len(image_paths_[40:50])
    return image_paths_flat, labels_flat


def preprocess_image_batch(image_batch, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []
    for img in image_batch:
        if img_size:
            img = imresize(img,img_size)
        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        # if color_mode=="bgr":
        #    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        if crop_size:
            img = img[(img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2, (img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2, :];
        img_list.append(img)
    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch

def undo_image_avg(img):
    img_copy = np.copy(img)
    img_copy[:, :, 0] = img_copy[:, :, 0] + 123.68
    img_copy[:, :, 1] = img_copy[:, :, 1] + 116.779
    img_copy[:, :, 2] = img_copy[:, :, 2] + 103.939
    return img_copy


def save_variables_and_metagraph(sess, saver, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
