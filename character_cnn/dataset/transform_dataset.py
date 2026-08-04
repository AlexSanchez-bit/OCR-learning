from torchvision import transforms

def dataset_transform():
    return transforms.Compose([
        transforms.RandomRotation(10), #rotates the image on a radom between -10 10 degree
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),# move the image (affine movement) with no rotation and 10% transalation
        transforms.ToTensor(), #<- transforms image data into tensor
        transforms.Normalize(mean=(0.5,), std=(0.5,)), #<- normalize tensor values
    ])
