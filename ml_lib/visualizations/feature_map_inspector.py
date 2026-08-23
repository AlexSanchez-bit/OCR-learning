from ml_lib.visualizations.feature_inspector import IFeatureInspector
import torch
from typing import Optional,Callable


class FeatureMapInspector(IFeatureInspector):

    def __init__(self):
        super().__init__()
        self.observerd={}
        self.network_map={}

    def watch(self,name:str,source:torch.nn.Module,handler:Optional[Callable[[torch.Tensor],torch.Tensor]]=None):
        def hook(module,inputs,output):
            if handler != None:
                self.network_map[name]=handler(output.detach())
            else:
                self.network_map[name]=output.detach()
        hookref=source.register_forward_hook(hook)
        self.observerd[name]=hookref

    def watch_live(self,name:str,source:torch.nn.Module,handler:Optional[Callable[[torch.Tensor],torch.Tensor]]=None):
        def hook(module,inputs,output):
            if handler != None:
                self.network_map[name]=handler(output)
            else:
                self.network_map[name]=output
        hookref=source.register_forward_hook(hook)
        self.observerd[name]=hookref

    def detach(self,name:str):
        handle = self.observerd.pop(name, None)
        if handle is not None:
                handle.remove()

    def get_data(self,name)->torch.Tensor:
        return self.network_map[name]
