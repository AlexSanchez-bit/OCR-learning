import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from typing import List
from ml_lib.checkpoint.checkpoint_saver import CheckpointSaver
from ml_lib.trainer.history import History,HistoryItem



class Trainer:
    def __init__(self,model:nn.Module,
                 train_loader:DataLoader,
                 validation_loader:DataLoader,
                 criterion,
                 optimizer,
                 device,
                 accuracy_calc,
                 checkpoint_manager:CheckpointSaver
                 ) -> None:
        self.model=model
        self.train_loader=train_loader
        self.validation_loader=validation_loader
        self.criterion=criterion
        self.optimizer=optimizer
        self.device=device
        self.accuracy_calc=accuracy_calc
        self.history=History()
        self.checkpoint_manager=checkpoint_manager

        self.model.to(device)
#move optimizer parameters to same device as model
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    def train_epoch(self)->HistoryItem:
        self.model.train() # set the model in train mode
        epoch_item = HistoryItem('train',0,0,[],[])
        for  batch_images,batch_labels in self.train_loader:
            # pass data into same device as model is
            batch_images = batch_images.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.optimizer.zero_grad() # reset gradients to avoid gradient accumulation
            output = self.model(batch_images) # forward passs on batch images
            loss=self.criterion(output,batch_labels)# loss function
            loss.backward()# backpropagation  (backward pass)
            self.optimizer.step()# weights update
            epoch_item.loss+=loss.item()
            epoch_item.accuracy += self.accuracy_calc(output,batch_labels)
            epoch_item.model_outputs.append(output.detach())
            epoch_item.expected_values.append(batch_labels.detach())

        total_images = len(self.validation_loader)
        epoch_item.loss/=total_images
        epoch_item.accuracy /= total_images

        return epoch_item

    def validation_epoch(self)->HistoryItem:
        self.model.eval() # set the model in evaluation mode
        epoch_item = HistoryItem('validation',0,0,[],[])
        with torch.no_grad():
          for  batch_images,batch_labels in self.validation_loader:
              # pass data into same device as model is
              batch_images = batch_images.to(self.device)
              batch_labels = batch_labels.to(self.device)
              output = self.model(batch_images) # forward passs on batch images
              loss=self.criterion(output,batch_labels)# loss function
              epoch_item.loss+=loss.item()
              epoch_item.accuracy += self.accuracy_calc(output,batch_labels)
              epoch_item.model_outputs.append(output.detach())
              epoch_item.expected_values.append(batch_labels.detach())

        total_images = len(self.validation_loader)
        epoch_item.loss/=total_images
        epoch_item.accuracy /= total_images
        return epoch_item

    def train(self,epoch_count:int,callback):
        for i in range(epoch_count):
            train_hi=self.train_epoch()
            self.history.register(train_hi)
            callback(train_hi)
            validation_hi=self.validation_epoch()
            self.history.register(validation_hi)
            callback(validation_hi)
            print(f"epoch {i+1}/{epoch_count}")
            self.checkpoint_manager.compare_and_save(i+1,validation_hi.loss,validation_hi.accuracy,self.model,self.optimizer)


