from torchvision.models import resnet18
from coreOperator import get_params, get_flops


model = resnet18()
print(get_params(model), get_flops(model))
