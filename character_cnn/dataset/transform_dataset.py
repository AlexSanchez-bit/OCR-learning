from torchvision import transforms

def dataset_transform():
    return transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0,0),
             scale=(0.3, 1),  # scale the content with p
            fill=0 ),# move the image (affine movement) with no rotation and 20% transalation
        transforms.RandomRotation(10), #rotates the image on a radom between -30 30 degree
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),# move the image (affine movement) with no rotation and 20% transalation
        transforms.ToTensor(), #<- transforms image data into tensor
        transforms.Normalize(mean=(0.5,), std=(0.5,)), #<- normalize tensor values
    ])
