from abc import ABC,abstractmethod
import torch
from typing import List

class ConvolutionPresenter(ABC):
  
  @abstractmethod
  def get_convolutions(self)->List[torch.Tensor]:
      pass

