import numpy as np
import cv2
from skimage.measure import compare_ssim
from asian_model import LCA
import torch 
import torch.nn as nn
from tools.hooks import *

handler_collection = []

def Unit_Conversion(num, types=None):
    if types == None:
        return num
    if(types=='G'):
        num = num/1e9
    if(types=='M'):
        num = num/1e6
    if(types=='K'):
        num = num/1e3
    return num 

def flops_route(module):
    if isinstance(module, nn.Conv2d):
        return conv2d_flops
    elif isinstance(module, nn.BatchNorm2d):
        return bn2d_flops
    elif isinstance(module, (nn.ReLU, nn.PReLU)):
        return relu_flops
    elif isinstance(module, nn.Linear):
        return linear_flops
    else:
        print("coming soon...")
        return None
                
def register_hooks(module):
    if(len(list(module.children())))>0:
        return 
    module.register_buffer('FLOPs', torch.zeros(1).long())
    module_type = type(module)
    fn = flops_route(module_type)
    if fn is not None:
        handler = module.register_forward_hook(fn)
        handler_collection.append(handler)
    else:
        print('None')

def get_params(net):
    count = 0
    for param in net.parameters():
        count += param.data.nelement()
    return Unit_Conversion(count)

def get_flops(net, input_size=(1,3,64,64),types='G'):
    net.apply(register_hooks)
    inputs = torch.Tensor(torch.zeros(input_size))
    with torch.no_grad():
        net(inputs)
    total_ops = 0
    for module in net.modules():
        if len(list(module.children()))>0:
            continue 
        total_ops += module.FLOPs.item()

    for handler in handler_collection:
        handler.remove()
    return  Unit_Conversion(total_ops, types)

