from abc import ABC,abstractmethod
from typing import Optional

class Checkpoint:
    def __init__(self,epoch,model,optimizer):
        self.epoch = epoch
        self.model=model
        self.optimizer=optimizer

    def get_data(self):
        return {
          "epoch":self.epoch,
          "model":self.model,
          "optimizer":self.optimizer,
          }

      
class CheckpointLoader(ABC):
      @abstractmethod
      def load_checkpoint_data(self)->Optional[Checkpoint]:
          """loads the model checkpoint"""
          pass

class CheckpointSaver(ABC):

    def compare_and_save(self,epoch,loss,accuracy,model,optimizer)->None:
      if self._save_condition(loss,accuracy):
        checkpoint= Checkpoint(epoch,model,optimizer)
        self._save_checkpoint(checkpoint)

    @abstractmethod
    def _save_checkpoint(self,checkpoint:Checkpoint)->None:
      """saves the current checkpoint"""
      pass

    @abstractmethod
    def _save_condition(self,loss,accuracy)->bool:
      """validates if the state is valid to save"""
      pass

