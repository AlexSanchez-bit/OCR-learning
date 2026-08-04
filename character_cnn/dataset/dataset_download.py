from torchvision import datasets


def load_dataset(transform,is_train):
# characters dataset
    return datasets.EMNIST(
        root="character_cnn/data", # <- folder to save and read from data
        split="balanced", #<- biased split (neutral this case)
        train=is_train, # <- training purpose
        download=True, # <- download the dataset
        transform=transform
    )


