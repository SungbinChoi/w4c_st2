import random
from random import shuffle
import numpy as np
from datetime import datetime
import time
import queue
import threading
import logging
from PIL import Image
import itertools
import re
import os
import glob
import shutil
import sys
import copy
import h5py
from netCDF4 import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from typing import Any, List, Tuple

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



input_dir_base_list = [
                       'a1/output',
                       'a1/output_aug1',
                       'a1/output_aug2',
                       'a1/output_aug3',
                       'a1/output_aug4',
                       'a1/output_aug5',
                       
                       'a2/output',
                       'a2/output_aug1',
                       'a2/output_aug2',
                       'a2/output_aug3',
                       'a2/output_aug4',
                       'a2/output_aug5',
                       
                       'a3/output',
                       'a3/output_aug1',
                       'a3/output_aug2',
                       'a3/output_aug3',
                       'a3/output_aug4',
                       'a3/output_aug5',
                       
                       'a4/output',
                       'a4/output_aug1',
                       'a4/output_aug2',
                       'a4/output_aug3',
                       'a4/output_aug4',
                       'a4/output_aug5',
                       
                       'b1/output',
                       'b1/output_aug1',
                       'b1/output_aug2',
                       'b1/output_aug3',
                       'b1/output_aug4',
                       'b1/output_aug5',
                       
                       'b2/output',
                       'b2/output_aug1',
                       'b2/output_aug2',
                       'b2/output_aug3',
                       'b2/output_aug4',
                       'b2/output_aug5',
                       
                       'b3/output',
                       'b3/output_aug1',
                       'b3/output_aug2',
                       'b3/output_aug3',
                       'b3/output_aug4',
                       'b3/output_aug5',
                       
                      ]
input_dir_base_target_var_dict = dict()

input_dir_base_target_var_dict['a1/output']      = [0,1,2,]
input_dir_base_target_var_dict['a1/output_aug1'] = [0,1,2,]
input_dir_base_target_var_dict['a1/output_aug2'] = [0,1,2,]
input_dir_base_target_var_dict['a1/output_aug3'] = [0,1,2,]
input_dir_base_target_var_dict['a1/output_aug4'] = [0,1,2,]
input_dir_base_target_var_dict['a1/output_aug5'] = [0,1,2,]

input_dir_base_target_var_dict['a2/output']      = [0,1,2,]
input_dir_base_target_var_dict['a2/output_aug1'] = [0,1,2,]
input_dir_base_target_var_dict['a2/output_aug2'] = [0,1,2,]
input_dir_base_target_var_dict['a2/output_aug3'] = [0,1,2,]
input_dir_base_target_var_dict['a2/output_aug4'] = [0,1,2,]
input_dir_base_target_var_dict['a2/output_aug5'] = [0,1,2,]

input_dir_base_target_var_dict['a3/output']      = [0,1,2,]
input_dir_base_target_var_dict['a3/output_aug1'] = [0,1,2,]
input_dir_base_target_var_dict['a3/output_aug2'] = [0,1,2,]
input_dir_base_target_var_dict['a3/output_aug3'] = [0,1,2,]
input_dir_base_target_var_dict['a3/output_aug4'] = [0,1,2,]
input_dir_base_target_var_dict['a3/output_aug5'] = [0,1,2,]

input_dir_base_target_var_dict['a4/output']      = [0,1,2,]
input_dir_base_target_var_dict['a4/output_aug1'] = [0,1,2,]
input_dir_base_target_var_dict['a4/output_aug2'] = [0,1,2,]
input_dir_base_target_var_dict['a4/output_aug3'] = [0,1,2,]
input_dir_base_target_var_dict['a4/output_aug4'] = [0,1,2,]
input_dir_base_target_var_dict['a4/output_aug5'] = [0,1,2,]

input_dir_base_target_var_dict['b1/output']      = [3,]
input_dir_base_target_var_dict['b1/output_aug1'] = [3,]
input_dir_base_target_var_dict['b1/output_aug2'] = [3,]
input_dir_base_target_var_dict['b1/output_aug3'] = [3,]
input_dir_base_target_var_dict['b1/output_aug4'] = [3,]
input_dir_base_target_var_dict['b1/output_aug5'] = [3,]

input_dir_base_target_var_dict['b2/output']      = [3,]
input_dir_base_target_var_dict['b2/output_aug1'] = [3,]
input_dir_base_target_var_dict['b2/output_aug2'] = [3,]
input_dir_base_target_var_dict['b2/output_aug3'] = [3,]
input_dir_base_target_var_dict['b2/output_aug4'] = [3,]
input_dir_base_target_var_dict['b2/output_aug5'] = [3,]

input_dir_base_target_var_dict['b3/output']      = [3,]
input_dir_base_target_var_dict['b3/output_aug1'] = [3,]
input_dir_base_target_var_dict['b3/output_aug2'] = [3,]
input_dir_base_target_var_dict['b3/output_aug3'] = [3,]
input_dir_base_target_var_dict['b3/output_aug4'] = [3,]
input_dir_base_target_var_dict['b3/output_aug5'] = [3,]


target_city_list  = ['R1', 'R2', 'R3', 'R7', 'R8',   
                     'R4', 'R5', 'R6', 'R9', 'R10', 'R11']
out_dir_base_list = ['submit', 'submit', 'submit', 'submit', 'submit',   
                     'submit2', 'submit2', 'submit2', 'submit2', 'submit2', 'submit2', ]
num_test_days = 36
num_frame_per_day  = 96 
num_frame_before   =  4     
num_frame_out      = 32     
num_frame_sequence = 36

height=256
width =256
num_channel_out= 4 
SEED = 0
EPS = 1e-12
np.set_printoptions(precision=6)

num_prediction_per_target_list = np.zeros((num_channel_out), np.float32)
for input_dir_base in input_dir_base_list:
  target_c_list = input_dir_base_target_var_dict[input_dir_base]
  for c in target_c_list:
    num_prediction_per_target_list[c] += 1
        
def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data=data, dtype=np.uint16, compression='gzip', compression_opts=9)
    f.close()



if __name__ == '__main__':
  COMMON_STRING ='@%s:  \n' % os.path.basename(__file__)
  COMMON_STRING += '\tset random seed\n'
  COMMON_STRING += '\t\tSEED = %d\n'%SEED
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)
  torch.backends.cudnn.enabled       = True
  torch.backends.cudnn.benchmark     = True
  torch.backends.cudnn.deterministic = True

  prediction = np.zeros((num_channel_out, num_frame_out, height, width), np.float32)
  for target_city_i, target_city in enumerate(target_city_list):
    out_dir_base = out_dir_base_list[target_city_i]
    out_dir = out_dir_base + '/' + target_city + '/' + 'test'
    try:
              if not os.path.exists(out_dir):
                  os.makedirs(out_dir)
    except Exception:
              print('out_dir not made')
              exit(-1)

    prediction_filename_list = []
    for prediction_filename in os.listdir(input_dir_base_list[0] + '/' + target_city):
      prediction_filename_list.append(prediction_filename)
    assert len(prediction_filename_list) == num_test_days

    for t, prediction_filename in enumerate(prediction_filename_list):
      day_name = prediction_filename.split('.')[0]
      out_file_path = os.path.join(out_dir, day_name + '.h5')
      
      prediction[:,:,:,:] = 0
      for input_dir_base in input_dir_base_list:
        prediction_one = np.load(input_dir_base + '/' + target_city + '/' + prediction_filename)['prediction']
        for c in range(num_channel_out):
          if c in input_dir_base_target_var_dict[input_dir_base]:
            prediction[c,:,:,:] += (prediction_one[0,c,:,:,:])

      for c in range(num_channel_out):
        prediction[c,:,:,:] /= num_prediction_per_target_list[c]
      pred_out = np.moveaxis(prediction, 0, 1)
      pred_out[:,0,:,:] *= 22000
      pred_out[:,1,:,:] *= 500 
      pred_out[:,2,:,:] *= 100
      pred_out_binary = pred_out[:,3,:,:].copy()
      pred_out_binary[pred_out_binary>0.5]  = 1
      pred_out_binary[pred_out_binary<=0.5] = 0
      pred_out[:,3,:,:] = pred_out_binary 
      pred_out = np.rint(pred_out)    
      pred_out = pred_out.astype(np.uint16)  
      write_data(pred_out, out_file_path)
    print(target_city)

  exit(1)
  
  