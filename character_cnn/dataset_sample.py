from dataset.data_loaders import get_character_loaders
from ml_lib.visualizations.data_loader_visualization import visualize_data_loader_batch 
from ml_lib.visualizations.dataset_visualization import show_dataset_metadata 
from indexation.save_indexed_chars import load_from_json

train_loader,_=get_character_loaders()

classes_map = load_from_json()
classes = list(map(lambda x:int(x),classes_map.keys()))
classes_labels = list(classes_map.values())
visualize_data_loader_batch(train_loader,train_loader.dataset.classes)
