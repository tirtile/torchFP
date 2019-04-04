import numpy as np
import cv2
from skimage.measure import compare_ssim
from asian_model import LCA
import torch 
import torch.nn as nn

#-------------------------------
# input feature shape : BxCxHxW
# output feature shape: BxCxHxW 
# FLOPs = (2Ci x K x K - 1) x H x W x Co
#       = ((Ci x K x K) + (Ci x K x K -1))x H x W x Co(without bias)
# Via: PRUNING CONVOLUTIONAL NEURAL NETWORKS FOR RESOURCE EFFICIENT INFERENCE, ICLR2017
#-------------------------------
def conv2d_flops(module, input_feature, output_feature):
    batch_size, inc = len(input_feature), module.in_channels
    _, out_c, out_h, out_w = output_feature.size(0),output_feature.size(1),output_feature.size(2),output_feature.size(3)
    k_h, k_w = module.kernel_size
    group = module.groups
    pixel_ops = k_h * k_w
    if module.bias is not None:
        pixel_ops+=1
    pixel_counts = out_c * out_h * out_w
    total_ops = pixel_ops * inc * pixel_counts  // group 
    module.FLOPs = torch.Tensor([int(total_ops)])

#----------------
# batchnorm2d: the same channel of images in a batch has the same mean and variance 
# 4 ops: sub, div(for mean_variance and alpha_beta)
#----------------
def bn2d_flops(module, input_feature, output_feature):
    batch_size, out_c, out_h, out_w = len(input_feature[0]), len(input_feature[0][0]),len(input_feature[0][0][0]),len(input_feature[0][0][0][0])
    batch_flops = out_c * out_h * out_w 
    total_ops = batch_flops * 4
    module.FLOPs = torch.Tensor([int(total_ops)])

#----------------
# relu: a operation
# x > 0=> x = x 
# x < 0=> x = x*0
#----------------  
def relu_flops(module, input_feature, output_feature):
    batch_size, out_c, out_h, out_w = len(input_feature[0]), len(input_feature[0][0]),len(input_feature[0][0][0]),len(input_feature[0][0][0][0])
    total_ops = out_c *  out_h * out_w 
    module.FLOPs = torch.Tensor([int(total_ops)])


#----------------
# relu: a operation
# x > 0=> x = x 
# x < 0=> x = x*0
#----------------  
def maxpool_flops(module, input_feature, output_feature):
    k_h, k_w = module.kernel_size, module.kernel_size
    _, out_c, out_h, out_w = output_feature.shape
    k_ops = k_w * k_h
    total_ops = k_ops * (out_c * out_w *out_h)
    module.FLOPs = torch.Tensor([int(total_ops)])


def softmax_flops(module, input_feature, output_feature):
    batch_size, units = input_feature.size()
    exp = units 
    add = units 
    div = units 
    total_ops = batch_size * (exp + add + div)
    module.FLOPs = torch.Tensor([int(total_ops)])

#----------------
# linear_flops 
# x_ij = sum(a_i-1 * w) 
#----------------  
def linear_flops(module, input_feature, output_feature):
    in_feature = input_feature.size(0)
    mul_ops = in_feature
    add_ops = in_feature-1
    total_ops = (mul_ops + add_ops)*(output_feature.size()[1])
    module.FLOPs = torch.Tensor([int(total_ops)])
