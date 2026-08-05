from models.character_model import CharacterModel
import torch
import threading
from ml_lib.trainer.trainer import Trainer
from dataset.data_loaders import get_character_loaders

from indexation.save_indexed_chars import load_from_json

from ml_lib.visualizations.real_time_train_visualization import RealTimeDataObserver
from ml_lib.checkpoint.weighted_checkpoint_saver import WeightedCheckpointSaver
from constants import MODEL_STATE_PATH


train_loader,validation_loader=get_character_loaders()

checkpoint_manager = WeightedCheckpointSaver(-0.3,1.7,MODEL_STATE_PATH)

last_state = checkpoint_manager.load_checkpoint_data()

model = CharacterModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

if last_state != None:
    print('loaded model state')
    model.load_state_dict(last_state.model)
    optimizer.load_state_dict(last_state.optimizer)


loss_f = torch.nn.CrossEntropyLoss()

def accuracy_calc(data,labels):
    activations=data.argmax(dim=1)#argmax will retrieve the index with the higher value (most likable class)
    return (activations==labels).sum().item()

observer = RealTimeDataObserver()

trainer = Trainer(model,train_loader,validation_loader,loss_f,optimizer,'cuda',accuracy_calc,checkpoint_manager)
classes_map = load_from_json()
classes = list(map(lambda x:int(x),classes_map.keys()))
classes_labels = list(classes_map.values())

def send_data(data):
    if observer!=None:
        observer.observe(data)
t=threading.Thread(target=trainer.train,args=[200,send_data])
t.start()
observer.start_loss_graph()
t.join()
observer.confussion_matrix(trainer.history,classes,classes_labels)


