# torchFP
tools for FLOPs and parameters calculation of pytorch 

# How to Use ?
'''
from torchvision.models import resnet18
from coreOperator import get_params, get_flops

model = resnet18()
print(get_params(model), get_flops(model))
'''
