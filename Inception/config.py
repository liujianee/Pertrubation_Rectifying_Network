#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
  return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--kernel_dims', type=eval, default='[]', help='')
net_arg.add_argument('--stride_size', type=eval, default='[]', help='')
net_arg.add_argument('--channel_dims', type=eval, default='[]', help='')
net_arg.add_argument('--resnet_num', type=int, default=5)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_dir', type=str, default='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL/')
data_arg.add_argument('--data_dir_1', type=str, default='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL_Corner_1/')
data_arg.add_argument('--data_dir_2', type=str, default='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL_Corner_2/')
data_arg.add_argument('--data_dir_3', type=str, default='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL_Corner_3/')
data_arg.add_argument('--data_dir_4', type=str, default='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL_Corner_4/')
data_arg.add_argument('--input_height', type=int, default=224)
data_arg.add_argument('--input_width', type=int, default=224)
data_arg.add_argument('--input_channel', type=int, default=3)
data_arg.add_argument('--net_type', type=str, default='inception')
data_arg.add_argument('--perturb_dir', type=str, default='/media/jianl/TOSHIBA-EXT/Projects/Universal_Perturbations/python/')
#data_arg.add_argument('--pert_train_list', type=str, default=['pert_gen_1/universal_10000_0/universal_10000_0_itr_1.npy'])
data_arg.add_argument('--pert_train_dir', type=str, default='Perturbations_Inception_Linf_All/')
data_arg.add_argument('--pert_test_list', type=str,  default=['pert_gen_0/universal_10000_0/universal_10000_0_itr_1.npy'])

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--task', type=str, default='generative', 
                       choices=['generative', 'estimation'], help='')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='')
train_arg.add_argument('--max_step', type=int, default=3000, help='')
train_arg.add_argument('--reg_scale', type=float, default=0.0005, help='')
train_arg.add_argument('--batch_size', type=int, default=50, help='')
train_arg.add_argument('--num_epochs', type=int, default=12, help='')
train_arg.add_argument('--random_seed', type=int, default=123, help='')
train_arg.add_argument('--learning_rate', type=float, default=0.001, help='')
train_arg.add_argument('--checkpoint_secs', type=int, default=300, help='')
train_arg.add_argument('--max_grad_norm', type=float, default=50, help='')
train_arg.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'], help='')
train_arg.add_argument('--nnclassifier_pretrained_model', type=str, default='/media/jianl/TOSHIBA-EXT/Projects/Universal_Perturbations/python/data/tensorflow_inception_graph.pb', help='')
train_arg.add_argument('--pretrained_model', type=str, default='', help='')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_step', type=int, default=2, help='')
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--mdl_dir', type=str, default='models')
misc_arg.add_argument('--sample_dir', type=str, default='samples')
misc_arg.add_argument('--output_dir', type=str, default='outputs')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--debug', type=str2bool, default=False)
misc_arg.add_argument('--gpu_memory_fraction', type=float, default=1.0)
misc_arg.add_argument('--max_image_summary', type=int, default=5)
misc_arg.add_argument('--sample_image_grid', type=eval, default='[3, 3]')

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed
