from torchvision import transforms
import random
from PIL import ImageOps

def dataset_transform():

    def random_invert(img):
        if random.random() > 0.5:
            return ImageOps.invert(img) 
        return img


    return transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0,0),
             scale=(0.3, 1.0),  # scale the content with p
            fill=0 ),# move the image (affine movement) with no rotation and 20% transalation
        transforms.RandomRotation(20), #rotates the image on a radom between -30 30 degree
        transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),# move the image (affine movement) with no rotation and 20% transalation
        transforms.Lambda(random_invert),# invert background to void background pattern dependency
        transforms.ToTensor(), #<- transforms image data into tensor
        transforms.Normalize(mean=(0.5,), std=(0.5,)), #<- normalize tensor values
    ])
