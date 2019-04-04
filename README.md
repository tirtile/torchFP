# torchFP
tools for FLOPs and parameters calculation of pytorch 

# How to Use ?
```
from torchvision.models import resnet18
from coreOperator import get_params, get_flops

model = resnet18()
print(get_params(model,input_size=(1,3,64,64),types="G"), get_flops(model, types="G"))
```
