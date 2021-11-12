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

out_dir_base = 'output'
model_checkpoint     = './model.pth'
target_city_list = ['R1', 'R2', 'R3', 'R7', 'R8',   'R4', 'R5', 'R6', 'R9', 'R10', 'R11']
input_data_folder_path_base    = '../..'
aug_method_list = ['aug1', 'aug2', 'aug3', 'aug4', 'aug5',]
num_frame_per_day  = 96
num_frame_before   =  4        
num_frame_out      = 32      
num_frame_sequence = 36
height=256
width =256
num_channel_1 = 9    
num_channel_2_src = 16 
num_channel_2 = 107 + num_channel_2_src
num_channel = (num_channel_1*2 + num_channel_2)
num_channel_out= 4
NUM_INPUT_CHANNEL  = num_channel     * num_frame_before
NUM_OUTPUT_CHANNEL = num_channel_out * num_frame_out
input_data_2_normalizer = 4
SEED = 0
num_groups = 8
EPS = 1e-12
np.set_printoptions(precision=6)








def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data=data, dtype=np.uint16, compression='gzip', compression_opts=9)
    f.close()

class Deconv3x3Block(nn.Sequential):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, ) -> None:
        super(Deconv3x3Block, self).__init__()
        self.add_module('deconv', nn.ConvTranspose2d(in_size, h_size, kernel_size=3, stride=2, padding=1, bias=True))
        self.add_module('elu',  nn.ELU(inplace=True))                                        
        self.add_module('norm', nn.GroupNorm(num_groups=num_groups, num_channels=h_size))    

class Conv1x1Block(nn.Sequential):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, ) -> None:
        super(Conv1x1Block, self).__init__()
        self.add_module('conv', nn.Conv2d(in_size, h_size, kernel_size=1, stride=1, padding=0, bias=True))

class Conv3x3Block(nn.Sequential):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, ) -> None:
        super(Conv3x3Block, self).__init__()
        self.add_module('conv', nn.Conv2d(in_size, h_size, kernel_size=3, stride=1, padding=1, bias=True))
        self.add_module('elu',  nn.ELU(inplace=True))                                        
        self.add_module('norm', nn.GroupNorm(num_groups=num_groups, num_channels=h_size))    

class AvgBlock(nn.Sequential):
    def __init__(self, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int) -> None:
        super(AvgBlock, self).__init__()
        self.add_module('pool', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))    
        
class MaxBlock(nn.Sequential):
    def __init__(self, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int) -> None:
        super(MaxBlock, self).__init__()
        self.add_module('pool', nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))    

class DownBlock(nn.Module):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, 
                 out_size: int, 
                 do_pool: int = True):
        super(DownBlock, self).__init__()     

        self.do_pool = do_pool
        self.pool = None
        if self.do_pool:
          self.pool = AvgBlock(kernel_size=2, stride=2, padding=0)

        in_size_cum = in_size  
        self.conv_1 = Conv3x3Block( in_size=in_size_cum, h_size=h_size)
        in_size_cum += h_size
        self.conv_3 = Conv3x3Block( in_size=in_size_cum, h_size=h_size)
        in_size_cum += h_size
        self.conv_2 = Conv1x1Block( in_size=in_size_cum,  h_size=out_size)

    def forward(self, x):
        
        batch_size = len(x)
        if self.do_pool:
          x = self.pool(x)
        x_list = []
        x_list.append(x)
        x = self.conv_1(x)
        x_list.append(x)
        x = torch.cat(x_list, 1)
        x = self.conv_3(x)
        x_list.append(x)
        x = torch.cat(x_list, 1)
        x = self.conv_2(x)
        return x

    def cuda(self, ):
        super(DownBlock, self).cuda()   
        self.conv_1.cuda()
        self.conv_3.cuda()
        self.conv_2.cuda()
        return self
        

class UpBlock(nn.Module):
    def __init__(self, 
                 in_size:   int, 
                 in_size_2: int, 
                 h_size:    int, 
                 out_size:  int, 
                 ):
        super(UpBlock, self).__init__()     
        self.deconv   = Deconv3x3Block( in_size=in_size, h_size=h_size)
        self.out_conv = Conv3x3Block( in_size=h_size + in_size_2, h_size=out_size)

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        x1 = F.interpolate(x1, size=x2.size()[2:4], scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None)
        x = torch.cat([x2, x1], dim=1)
        return self.out_conv(x)

    def cuda(self, ):
        super(UpBlock, self).cuda()   
        self.deconv.cuda()
        self.out_conv.cuda()
        return self


class NetA(nn.Module):

    def __init__(self,):
        super(NetA, self).__init__()
        self.block0 = DownBlock(in_size=NUM_INPUT_CHANNEL, h_size=64, out_size=64, do_pool=False)
        self.block1 = DownBlock(in_size=64, h_size=96, out_size=96,)
        self.block2 = DownBlock(in_size=96, h_size=128, out_size=128, )
        self.block3 = DownBlock(in_size=128, h_size=128, out_size=128, )
        self.block4 = DownBlock(in_size=128, h_size=128, out_size=128, )
        self.block5 = DownBlock(in_size=128, h_size=128, out_size=128, )
        self.block6 = DownBlock(in_size=128, h_size=128, out_size=128,)
        self.block20 = Conv3x3Block(in_size=128, h_size=128)
        self.block15 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,)
        self.block14 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,) 
        self.block13 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,)
        self.block12 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,)
        self.block11 = UpBlock(in_size=128, in_size_2=96 , h_size=128,  out_size=128,) 
        self.block10 = UpBlock(in_size=128, in_size_2=64 , h_size=128,  out_size=128,) 
        self.out_conv  = nn.Sequential(nn.Conv2d(128*1, NUM_OUTPUT_CHANNEL, kernel_size=3, stride=1, padding=1, bias=True))
        
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                  nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                  nn.init.constant_(m.weight, 1)
                  nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                  nn.init.constant_(m.weight, 1)
                  nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                  nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = len(x)
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        x  = self.block20(x6)
        x  = self.block15(x, x5)
        x  = self.block14(x, x4)
        x  = self.block13(x, x3)
        x  = self.block12(x, x2)
        x  = self.block11(x, x1)
        x  = self.block10(x, x0)
        x  = self.out_conv(x)
        x = torch.reshape(x, (batch_size, num_channel_out, 1, num_frame_out, height, width))
        return x[:,0,:,:,:,:], x[:,1,:,:,:,:], x[:,2,:,:,:,:], x[:,3,:,:,:,:]

    def cuda(self, ):
        super(NetA, self).cuda()
        self.block0.cuda()
        self.block1.cuda()
        self.block2.cuda()
        self.block3.cuda()
        self.block4.cuda()
        self.block5.cuda()
        self.block6.cuda()
        self.block20.cuda()
        self.block15.cuda()
        self.block14.cuda()
        self.block13.cuda()
        self.block12.cuda()
        self.block11.cuda()
        self.block10.cuda()
        self.out_conv.cuda()
        return self  


continuous_data_info_list = np.zeros((num_channel_1, 3), np.float32)
if 1:
  continuous_data_info_filepath = os.path.join(input_data_folder_path_base + '/0_data/', 'continuous_data_info_all.txt')

  c=0
  with open(continuous_data_info_filepath) as info_file:
    content = info_file.readlines()
    for line in content:
      cols = line.strip().split('\t')
      d_min    = int(  cols[0])
      d_max    = int(  cols[1])
      d_avg    = float(cols[2])
      continuous_data_info_list[c,:] = (d_min,d_max,d_avg)
      c += 1  
continuous_data_info_list_min = continuous_data_info_list[np.newaxis,:, 0, np.newaxis,np.newaxis,]
continuous_data_info_list_max = continuous_data_info_list[np.newaxis,:, 1, np.newaxis,np.newaxis,]

continuous_output_info_list = np.zeros((3, 2), np.float32)    
continuous_output_info_list[0,:] = (130, 350)      
continuous_output_info_list[1,:] = (0,    50)      
continuous_output_info_list[2,:] = (0,   100)      
continuous_output_info_list = continuous_output_info_list[np.newaxis, :, :, np.newaxis,np.newaxis,]
  
discrete_data_info_list = np.zeros((num_channel_2_src, ), np.uint8)
if 1:
  discrete_data_info_filepath   = os.path.join(input_data_folder_path_base + '/0_data/', 'discrete_data_info.txt')
  c=0
  with open(discrete_data_info_filepath) as info_file:
    content = info_file.readlines()
    for line in content:
      cols = line.strip().split('\t')
      num_flag = int(cols[0])
      discrete_data_info_list[c] = (num_flag+1)
      c += 1
      
cum_num_flag_list = np.zeros((num_channel_2_src, 2), np.uint8)
cc = 0
for c in range(num_channel_2_src):
  cum_num_flag_list[c,0] = cc
  cc+=discrete_data_info_list[c]
  cum_num_flag_list[c,1] = cc
assert cc < 256     


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

  
  try:
            if not os.path.exists(out_dir_base):
                os.makedirs(out_dir_base)
  except Exception:
            print('out_dir_base not made')
            exit(-1)

  for aug_method in aug_method_list:
    out_dir_base_aug = out_dir_base + '_' + aug_method
    try:
            if not os.path.exists(out_dir_base_aug):
                os.makedirs(out_dir_base_aug)
    except Exception:
            print('out_dir_base_aug not made ', out_dir_base_aug)
            exit(-1)

  net = NetA().cuda()
  asii_logit_m = -torch.logit(torch.from_numpy(np.array(0.003,np.float32)).float().cuda())
  assert model_checkpoint is not None
  state_dict_0 = torch.load(model_checkpoint, map_location=lambda storage, loc: storage)
  net.load_state_dict(state_dict_0, strict=True)
  net.eval()  
  
  
  index_list2 = np.arange(num_frame_before * height * width)
  def get_data(input_data_1, input_data_2):

    input_data_out_3 = np.ones((num_frame_before,    num_channel_1,  height, width), np.float32)      
    input_data_out_1 = (input_data_1 - continuous_data_info_list_min) / (continuous_data_info_list_max - continuous_data_info_list_min)
    
    input_data_out_1[input_data_1==65535] = 0
    input_data_out_3[input_data_1==65535] = 0
    input_data_out_1 = np.moveaxis(input_data_out_1, 1, 0) 
    input_data_out_3 = np.moveaxis(input_data_out_3, 1, 0)  

    input_data_2 += 1
    input_data_2[input_data_2==256] = 0
    one_hot_list = np.zeros((num_frame_before * height * width, num_channel_2), np.uint8)
    for c in range(num_channel_2_src):
      one_hot_list[index_list2, cum_num_flag_list[c,0] + input_data_2[:,c,:,:].reshape(-1)] = 1
    input_data_out_2 = np.moveaxis(one_hot_list, -1, 0).reshape(num_channel_2,   num_frame_before,    height, width)                    
    input_data_out = np.concatenate([input_data_out_1, input_data_out_3, input_data_out_2], axis=0)
    input_data_out = input_data_out.reshape(-1, height, width)

    return input_data_out
  
  
  
  
  
  
  for target_city in target_city_list:

    input_data_folder_path = input_data_folder_path_base + '/0_data/' + target_city
    out_dir = out_dir_base + '/' + target_city

    try:
              if not os.path.exists(out_dir):
                  os.makedirs(out_dir)
    except Exception:
              print('out_dir not made')
              exit(-1)

    for aug_method in aug_method_list:
      out_dir_base_aug = out_dir_base + '_' + aug_method
      out_dir_aug = out_dir_base_aug + '/' + target_city
    
      try:
                if not os.path.exists(out_dir_aug):
                    os.makedirs(out_dir_aug)
      except Exception:
                print('out_dir_aug not made ', out_dir_aug)
                exit(-1)

    asii_frame_file_name_prefix     = 'S_NWC_ASII-TF_MSG4_Europe-VISIR_'
    asii_frame_file_name_prefix_len = len(asii_frame_file_name_prefix)
    
    input_folder_path = input_data_folder_path + '/' + 'test'
    num_day_done = 0
    
    for day_folder_name in os.listdir(input_folder_path):
      
      day_folder_path = os.path.join(input_folder_path, day_folder_name)
      if os.path.isdir(day_folder_path) == False:
        continue
      
      day = int(day_folder_name)
      frame_file_name_list = []
      for frame_file_name in os.listdir(os.path.join(day_folder_path, 'ASII')):
        if frame_file_name.split('.')[-1] != 'nc':
          continue
        
        assert frame_file_name[asii_frame_file_name_prefix_len-1] == '_'
        assert frame_file_name[asii_frame_file_name_prefix_len+8] == 'T'
        
        frame_file_name_list.append(frame_file_name)
        

      assert len(frame_file_name_list) == num_frame_before
      frame_file_name_list = sorted(frame_file_name_list)
      
      min_before = 0
      ymd_list = []
      for frame_file_name in frame_file_name_list:
        ymd    = int(frame_file_name[asii_frame_file_name_prefix_len    : (asii_frame_file_name_prefix_len+8)])
        hour   = int(frame_file_name[asii_frame_file_name_prefix_len+9  : (asii_frame_file_name_prefix_len+11)])  
        minute = int(frame_file_name[asii_frame_file_name_prefix_len+11 : (asii_frame_file_name_prefix_len+13)])  
        ymd_list.append((ymd, hour, minute))
        
        min_now = (ymd-20190000)*24*60 + hour*60 + minute
        assert min_before < min_now
        min_before = min_now
        
      
      file_list=[]       
      for (ymd, hour, minute) in ymd_list:
        file_list.append(\
        (os.path.join(input_folder_path, str(day), 'CTTH', 'S_NWC_CTTH_MSG4_Europe-VISIR_'    + str(ymd) + ('T%02d%02d%02d' % (hour, minute, 0)) + 'Z.nc'),
        os.path.join(input_folder_path, str(day), 'CRR',  'S_NWC_CRR_MSG4_Europe-VISIR_'     + str(ymd) + ('T%02d%02d%02d' % (hour, minute, 0)) + 'Z.nc'),
        os.path.join(input_folder_path, str(day), 'ASII', 'S_NWC_ASII-TF_MSG4_Europe-VISIR_' + str(ymd) + ('T%02d%02d%02d' % (hour, minute, 0)) + 'Z.nc'),
        os.path.join(input_folder_path, str(day), 'CMA',  'S_NWC_CMA_MSG4_Europe-VISIR_'     + str(ymd) + ('T%02d%02d%02d' % (hour, minute, 0)) + 'Z.nc'),
        os.path.join(input_folder_path, str(day), 'CT',   'S_NWC_CT_MSG4_Europe-VISIR_'      + str(ymd) + ('T%02d%02d%02d' % (hour, minute, 0)) + 'Z.nc'),
        ))
      
      
      file_list_2=[]       
      for (ymd, hour, minute) in ymd_list:
        file_list_2.append(\
        (os.path.join(input_folder_path, str(day), 'CTTH', 'S_NWC_CTTH_MSG2_Europe-VISIR_'    + str(ymd) + ('T%02d%02d%02d' % (hour, minute, 0)) + 'Z.nc'),
        os.path.join(input_folder_path, str(day), 'CRR',  'S_NWC_CRR_MSG2_Europe-VISIR_'     + str(ymd) + ('T%02d%02d%02d' % (hour, minute, 0)) + 'Z.nc'),
        os.path.join(input_folder_path, str(day), 'ASII', 'S_NWC_ASII-TF_MSG2_Europe-VISIR_' + str(ymd) + ('T%02d%02d%02d' % (hour, minute, 0)) + 'Z.nc'),
        os.path.join(input_folder_path, str(day), 'CMA',  'S_NWC_CMA_MSG2_Europe-VISIR_'     + str(ymd) + ('T%02d%02d%02d' % (hour, minute, 0)) + 'Z.nc'),
        os.path.join(input_folder_path, str(day), 'CT',   'S_NWC_CT_MSG2_Europe-VISIR_'      + str(ymd) + ('T%02d%02d%02d' % (hour, minute, 0)) + 'Z.nc'),
        ))
      
      
      

      out_data_1 = np.zeros((num_frame_before, num_channel_1,     height, width), np.uint16)
      out_data_2 = np.zeros((num_frame_before, num_channel_2_src, height, width), np.uint8)
      
      out_data_1[:,:,:,:] = 65535
      out_data_2[:,:,:,:] = 255
      
      
      for f, (filepath_1, filepath_2, filepath_3, filepath_4, filepath_5) in enumerate(file_list):

        c1 = 0
        c2 = 0
        
        if 1:
            filepath = filepath_1
            dtype = np.dtype(np.uint16)
            if os.path.exists(filepath) == False:
              filepath = file_list_2[f][0]
            assert os.path.exists(filepath)
          
            ds = Dataset(filepath, 'r')

            key_list = ['temperature',
                        'ctth_tempe',
                        'ctth_pres',
                        'ctth_alti',
                        'ctth_effectiv',
                        'ctth_method',
                        'ctth_quality',
                        'ishai_skt',
                        'ishai_quality',
                        ]
            assert len(key_list) == 9
            
            for key in key_list:

              d_arr = np.array(ds[key]).astype(dtype)

              has_flag = False
              flags    = None
              try:
                flags = np.array(ds[key].flag_values, dtype)
                has_flag = True
              except:
                pass
                

              if has_flag:
                
                for flag_i, flag in enumerate(flags):
                  out_data_2[f, c2, d_arr == flag] = flag_i
                c2 += 1
                
              else:
                has_scale  = False
                has_offset = False
                scale_factor = 1.0
                offset       = 0.0
                
                try:
                  scale_factor = ds[key].scale_factor
                  has_scale    = True
                except:
                  pass
                try:
                  offset       = ds[key].add_offset
                  has_offset   = True
                except:
                  pass
                assert scale_factor != 0

                has_valid_range   = False
                valid_min         = 0
                valid_max         = 0
                try:
                  valid_min = ds[key].valid_range[0]
                  valid_max = ds[key].valid_range[1]
                  has_valid_range = True
                except:
                  pass

                mask_arr = np.ones((height, width), np.uint8)
                mask_arr[d_arr == ds[key]._FillValue] = 0                
                
                d_arr_float = d_arr.astype(np.float32)
                if has_offset:
                  d_arr_float -= offset
                if has_scale:
                  d_arr_float /= scale_factor
                
                if has_valid_range:
                  mask_arr[d_arr_float < valid_min] = 0   
                  mask_arr[d_arr_float > valid_max] = 0

                is_valid    = (mask_arr == 1)  
                is_invalid  = (mask_arr == 0)

                out_data_1[f, c1, :, :] = d_arr   
                out_data_1[f, c1, is_invalid]  = 65535
                
                c1 += 1
    


        if 1:
            filepath = filepath_2
            dtype = np.dtype(np.uint16)
            if os.path.exists(filepath) == False:
              filepath = file_list_2[f][1]
            assert os.path.exists(filepath)
            
            ds = Dataset(filepath, 'r')

            key_list = ['crr',
                        'crr_intensity',
                        'crr_accum',
                        'crr_quality',
                        ]
            assert len(key_list) == 4
            
            for key in key_list:
              
              d_arr = np.array(ds[key]).astype(dtype)
              
              has_flag = False
              flags    = None
              try:
                flags = np.array(ds[key].flag_values, dtype)
                has_flag = True
              except:
                pass

              if has_flag:
                for flag_i, flag in enumerate(flags):
                  out_data_2[f, c2, d_arr == flag] = flag_i
                c2 += 1
              else:
                has_scale  = False
                has_offset = False
                scale_factor = 1.0
                offset       = 0.0

                try:
                  scale_factor = ds[key].scale_factor
                  has_scale    = True
                except:
                  pass
                try:
                  offset       = ds[key].add_offset
                  has_offset   = True
                except:
                  pass

                has_valid_range   = False
                valid_min         = 0
                valid_max         = 0
                try:
                  valid_min = ds[key].valid_range[0]
                  valid_max = ds[key].valid_range[1]
                  has_valid_range = True
                except:
                  pass
                
                
                mask_arr = np.ones((height, width), np.uint8)
                mask_arr[d_arr == ds[key]._FillValue] = 0                 
                
                d_arr_float = d_arr.astype(np.float32)
                if has_offset:
                  d_arr_float -= offset
                if has_scale:
                  d_arr_float /= scale_factor
                if has_valid_range:
                  mask_arr[d_arr_float < valid_min] = 0 
                  mask_arr[d_arr_float > valid_max] = 0

                is_valid    = (mask_arr == 1)  
                is_invalid  = (mask_arr == 0)

                out_data_1[f, c1, :, :] = d_arr   
                out_data_1[f, c1, is_invalid]  = 65535   
                
                c1 += 1
    

        if 1:
            filepath = filepath_3
            dtype = np.dtype(np.uint8)
            if os.path.exists(filepath) == False:
              filepath = file_list_2[f][2]
            assert os.path.exists(filepath)
            ds = Dataset(filepath, 'r')
            key_list = ['asii_turb_trop_prob',
                        'asiitf_quality',
                        ]
            assert len(key_list) == 2
            
            for key in key_list:
              d_arr = np.array(ds[key]).astype(dtype)

              has_flag = False
              flags    = None
              try:
                flags = np.array(ds[key].flag_values, dtype)
                has_flag = True
              except:
                pass
              if has_flag:
                for flag_i, flag in enumerate(flags):
                  out_data_2[f, c2, d_arr == flag] = flag_i
                c2 += 1

              else:
                has_scale  = False
                has_offset = False
                scale_factor = 1.0
                offset       = 0.0

                try:
                  scale_factor = ds[key].scale_factor
                  has_scale    = True
                except:
                  pass
                try:
                  offset       = ds[key].add_offset
                  has_offset   = True
                except:
                  pass
                assert scale_factor != 0

                has_valid_range   = False
                valid_min         = 0
                valid_max         = 0
                try:
                  valid_min = ds[key].valid_range[0]
                  valid_max = ds[key].valid_range[1]
                  has_valid_range = True
                except:
                  pass

                mask_arr = np.ones((height, width), np.uint8)
                mask_arr[d_arr == ds[key]._FillValue] = 0
                
                d_arr_float = d_arr.astype(np.float32)
                if has_offset:
                  d_arr_float -= offset
                if has_scale:
                  d_arr_float /= scale_factor
                if has_valid_range:
                  mask_arr[d_arr_float < valid_min] = 0                  
                  mask_arr[d_arr_float > valid_max] = 0

                is_valid    = (mask_arr == 1)  
                is_invalid  = (mask_arr == 0)

                out_data_1[f, c1, :, :] = d_arr  
                out_data_1[f, c1, is_invalid]  = 65535   
                
                c1 += 1
    
        
        if 1:
            filepath = filepath_4
            dtype = np.dtype(np.uint8)
            if os.path.exists(filepath) == False:
              filepath = file_list_2[f][3]
            assert os.path.exists(filepath)
            
            ds = Dataset(filepath, 'r')
            key_list = ['cma_cloudsnow',
                        'cma',
                        'cma_dust',
                        'cma_volcanic',
                        'cma_smoke',
                        'cma_quality',
                        ]

            for key in key_list:
              d_arr = np.array(ds[key]).astype(dtype)
              has_flag = False
              flags    = None
              try:
                flags = np.array(ds[key].flag_values, dtype)
                has_flag = True
              except:
                pass

              if has_flag:
                for flag_i, flag in enumerate(flags):
                  out_data_2[f, c2, d_arr == flag] = flag_i
                c2 += 1

              else:

                has_scale  = False
                has_offset = False
                scale_factor = 1.0
                offset       = 0.0
                #
                try:
                  scale_factor = ds[key].scale_factor
                  has_scale    = True
                except:
                  pass
                try:
                  offset       = ds[key].add_offset
                  has_offset   = True
                except:
                  pass
                assert scale_factor != 0

                has_valid_range   = False
                valid_min         = 0
                valid_max         = 0
                try:
                  valid_min = ds[key].valid_range[0]
                  valid_max = ds[key].valid_range[1]
                  has_valid_range = True
                except:
                  pass

                
                mask_arr = np.ones((height, width), np.uint8)
                mask_arr[d_arr == ds[key]._FillValue] = 0 

                d_arr_float = d_arr.astype(np.float32)
                if has_offset:
                  d_arr_float -= offset
                if has_scale:
                  d_arr_float /= scale_factor
                if has_valid_range:
                  mask_arr[d_arr_float < valid_min] = 0   
                  mask_arr[d_arr_float > valid_max] = 0
                is_valid    = (mask_arr == 1)  
                is_invalid  = (mask_arr == 0)
                out_data_1[f, c1, :, :] = d_arr   
                out_data_1[f, c1, is_invalid]  = 65535   
                c1 += 1
    
            
        if 1:
            filepath = filepath_5
            dtype = np.dtype(np.uint8)
            if os.path.exists(filepath) == False:
              filepath = file_list_2[f][4]
            assert os.path.exists(filepath)
        
            ds = Dataset(filepath, 'r')
            key_list = ['ct',
                        'ct_cumuliform',
                        'ct_multilayer',
                        'ct_quality',
                        ]
            
            for key in key_list:
              d_arr = np.array(ds[key]).astype(dtype)
              has_flag = False
              flags    = None
              try:
                flags = np.array(ds[key].flag_values, dtype)
                has_flag = True
              except:
                pass

              if has_flag:
                for flag_i, flag in enumerate(flags):
                  out_data_2[f, c2, d_arr == flag] = flag_i
                c2 += 1
              else:
                has_scale  = False
                has_offset = False
                scale_factor = 1.0
                offset       = 0.0
                try:
                  scale_factor = ds[key].scale_factor
                  has_scale    = True
                except:
                  pass
                try:
                  offset       = ds[key].add_offset
                  has_offset   = True
                except:
                  pass
                assert scale_factor != 0
                
                has_valid_range   = False
                valid_min         = 0
                valid_max         = 0
                try:
                  valid_min = ds[key].valid_range[0]
                  valid_max = ds[key].valid_range[1]
                  has_valid_range = True
                except:
                  pass

                mask_arr = np.ones((height, width), np.uint8)
                mask_arr[d_arr == ds[key]._FillValue] = 0   
                
                d_arr_float = d_arr.astype(np.float32)
                if has_offset:
                  d_arr_float -= offset
                if has_scale:
                  d_arr_float /= scale_factor
                if has_valid_range:
                  mask_arr[d_arr_float < valid_min] = 0
                  mask_arr[d_arr_float > valid_max] = 0
                is_valid    = (mask_arr == 1)  
                is_invalid  = (mask_arr == 0)
                out_data_1[f, c1, :, :] = d_arr   
                out_data_1[f, c1, is_invalid]  = 65535   
                c1 += 1

      test_input_data_one = get_data(out_data_1, out_data_2)

      with torch.no_grad():

       for a in range(6):   
        input = None
        if a ==0:
          input = torch.from_numpy(test_input_data_one[np.newaxis, :, :, :]).float().cuda() 
        elif a ==1:
          input = torch.from_numpy(test_input_data_one[np.newaxis, :, :, ::-1].copy()).float().cuda() 
        elif a ==2:
          input = torch.from_numpy(test_input_data_one[np.newaxis, :, ::-1, :].copy()).float().cuda() 
        elif a ==3:
          input = torch.from_numpy(test_input_data_one[np.newaxis, :, ::-1, ::-1].copy()).float().cuda() 
        elif a ==4:
          input = torch.from_numpy(np.moveaxis(test_input_data_one[np.newaxis, :, :, :], 2, -1).copy()).float().cuda() 
        elif a ==5:
          input = torch.from_numpy(np.moveaxis(test_input_data_one[np.newaxis, :, :, ::-1], 2, -1).copy()).float().cuda() 
          
        logit0, logit1, logit2, logit3 = net(input)
        logit = torch.cat(                 
                [
                  torch.sigmoid(logit0),  
                  torch.sigmoid(logit1),  
                  torch.sigmoid(torch.sigmoid(logit2) * (asii_logit_m*2) - asii_logit_m),
                  torch.sigmoid(logit3), 
                ],              1)
        prediction = logit.cpu().detach().numpy()

        if a ==1:
          prediction = prediction[:,:,:,:,::-1].copy()      
        elif a ==2:
          prediction = prediction[:,:,:,::-1,:].copy()          
        elif a ==3:
          prediction = prediction[:,:,:,::-1,::-1].copy()          
        elif a ==4:
          prediction = np.moveaxis(prediction, 3, -1).copy()          
        elif a ==5:
          prediction = np.moveaxis(prediction[:,:,:,::-1,:], 3, -1).copy()               

        if a == 0:
          np.savez_compressed(os.path.join(out_dir, day_folder_name), prediction=prediction)
        else:
          out_dir_base_aug = out_dir_base + '_' + aug_method_list[a-1]
          out_dir_aug = out_dir_base_aug + '/' + target_city
          np.savez_compressed(os.path.join(out_dir_aug, day_folder_name), prediction=prediction)

      num_day_done += 1

    print(target_city, '\t', 'num_day_done:',   num_day_done,   '\t', )

  exit(1)
