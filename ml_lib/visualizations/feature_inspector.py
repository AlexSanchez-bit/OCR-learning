from abc import ABC,abstractmethod
import torch
from typing import Optional,Callable


class IFeatureInspector(ABC):

    @abstractmethod
    def watch(self,name:str,source:torch.nn.Module,handler:Optional[Callable[[torch.Tensor],torch.Tensor]]):
        """attach a data source (detached from the module graph) to the watch pipeline"""
        pass

    @abstractmethod
    def watch_live(self,name:str,source:torch.nn.Module,handler:Optional[Callable[[torch.Tensor],torch.Tensor]]=None):
        """attach a data source grad (atached to the module graph) to the watch pipeline"""
        pass

    @abstractmethod
    def detach(self,name:str):
        """detatch a data source to the watch pipeline"""
        pass

    @abstractmethod
    def get_data(self,name)->torch.Tensor:
        """retrieves the data"""
        pass


