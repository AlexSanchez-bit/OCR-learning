from torch.utils.data import DataLoader
from .dataset_download import load_dataset
from .character_dataset import CharacterDataset
from .transform_dataset import dataset_transform



def get_character_loaders():

  transform = dataset_transform()
  train_dataset = CharacterDataset(load_dataset(transform,True))
  validation_dataset = CharacterDataset(load_dataset(transform,False))

  train_loader = DataLoader(
      train_dataset,
      batch_size=64,
      shuffle=True,
      num_workers=0
  )

  validation_loader = DataLoader(
      validation_dataset,
      batch_size=64,
      shuffle=True,
      num_workers=0
  )

  return train_loader,validation_loader
