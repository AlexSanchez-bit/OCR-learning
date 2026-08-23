
from torch import Tensor
from torch.nn.functional import relu
from PIL import Image


def grad_cam(layer_tensor:Tensor,img):
    if layer_tensor.grad ==None:
        raise Exception("No grad on target layer")
        return
    layer_tensor_grad=layer_tensor.grad.detach().cpu().squeeze(dim=0)
    layer_tensor=layer_tensor.detach().cpu().squeeze(dim=0)
    a_k= layer_tensor_grad.mean(dim=(1,2))
    return relu((layer_tensor * a_k.view(-1,1,1)).sum(dim=0))
