from ml_lib.checkpoint.checkpoint_saver import CheckpointSaver,Checkpoint
from ml_lib.checkpoint.checkpoint_loader import BaseCheckpointLoader
from typing import Optional
import torch


class WeightedCheckpointSaver(CheckpointSaver,BaseCheckpointLoader):

      def __init__(self,loss_weight:float,accuracy_weight:float,filepath:str):
          self.loss_weight = loss_weight
          self.accuracy_weight = accuracy_weight
          self.filepath=filepath
          self.best_puntuation = 0

      def _save_checkpoint(self,checkpoint:Checkpoint)->None:
          torch.save({
              "epoch": checkpoint.epoch,
              "model": checkpoint.model.state_dict(),
              "optimizer": checkpoint.optimizer.state_dict(),
          },self.filepath)

      def _save_condition(self,loss,accuracy)->bool:
          puntuation = self.loss_weight * loss + self.accuracy_weight*accuracy
          if puntuation > self.best_puntuation:
             self.best_puntuation=puntuation
             return True
          return False
