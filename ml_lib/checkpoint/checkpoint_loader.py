from ml_lib.checkpoint.checkpoint_saver import CheckpointSaver,Checkpoint,CheckpointLoader
from typing import Optional
import torch


class BaseCheckpointLoader(CheckpointLoader):

      def __init__(self,filepath:str):
          self.filepath=filepath

      def load_checkpoint_data(self)->Optional[Checkpoint]:
          try:
            data = torch.load(self.filepath)
            if data:
              return Checkpoint(
                    data["epoch"],
                    data["model"],
                    data["optimizer"]
              )
          except:
            return None

