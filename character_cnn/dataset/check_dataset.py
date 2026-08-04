# from check_pytorch import check_pytorch_version_and_cuda
# from dataset_visualization import show_dataset_metadata
from transform_dataset import dataset_transform
from torch.utils.data import DataLoader
from data_loader_visualization import visualize_data_loader_batch
from save_indexed_chars import save_as_json

from dataset_download import load_dataset
from dataset.character_dataset import CharacterDataset

# check_pytorch_version_and_cuda()
transform = dataset_transform()
# without transformation images can be visualized tensors are not images
# train_dataset = load_dataset(None,True)
# show_dataset_metadata(train_dataset)
train_dataset = CharacterDataset(load_dataset(transform,True))
test_dataset = CharacterDataset(load_dataset(transform,False)) #test dataset

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)


#maps classes to labels
idx_to_char = {i: c for i, c in enumerate(train_dataset.classes)}
save_as_json(idx_to_char)
#chars to index
visualize_data_loader_batch(test_loader,idx_to_char)

images, labels = next(iter(train_loader))

print("Batch OK: ",images.shape)
print("Min:", images.min().item())
print("Max:", images.max().item())
print("example class:", idx_to_char[labels[0].item()])
