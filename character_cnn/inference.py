from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import random

from character_cnn.constants import MODEL_STATE_PATH
from character_cnn.indexation.save_indexed_chars import load_from_json
from character_cnn.models.character_model import CharacterModel
from ml_lib.checkpoint.checkpoint_loader import BaseCheckpointLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF



class ModelInference():
    def __init__(self):
        self.classes_map = load_from_json()
        checkpoint_manager = BaseCheckpointLoader(MODEL_STATE_PATH)
        last_state = checkpoint_manager.load_checkpoint_data()
        self.model = CharacterModel()
        if last_state is not None:
            self.model.load_state_dict(last_state.model)
        else:
            raise Exception("there's no model state to load, train first")
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])

    def apply_convolution(self,normalized_image,weights,bias):
        conv1 = F.conv2d(normalized_image, weights, bias=bias, stride=self.model.conv1.stride, padding=self.model.conv1.padding)
        return conv1

    def apply_rotation(self,img):
        img = TF.rotate(img, angle=90)
        img = TF.vflip(img)
        return img


    def get_filters(self,img,rows:int=4,columns:int=8):
        conv1_weights = self.model.conv1.weight.data.clone()
        conv2_weights = self.model.conv2.weight.data.clone()
        bias_conv1 = self.model.conv1.bias.detach().cpu() if self.model.conv1.bias is not None else None
        bias_conv2 = self.model.conv2.bias.detach().cpu() if self.model.conv1.bias is not None else None
        normalized_image = self.preprocess(img).unsqueeze(0)

        conv1 = self.apply_convolution(normalized_image,conv1_weights,bias_conv1)
        conv2 = self.apply_convolution(conv1,conv2_weights,bias_conv2)

        conv1=self.apply_rotation(conv1)
        conv2=self.apply_rotation(conv2)

        fig, axes = plt.subplots(rows, columns, figsize=(12, 4))
        for i in range(rows):
            for j in range(columns):
                if i < 2:
                    index = random.randint(0,conv1_weights.shape[0]-1)
                    img = conv1[0,index]
                else:
                    index = random.randint(0,conv2_weights.shape[0]-1)
                    img = conv2[0,index]
                img = img * 0.5 + 0.5  # remove normalization (inverse operation)

                axes[i,j].imshow(img, cmap="gray")
                axes[i,j].axis("off")
        return fig,axes



    def inference(self,img):
        x = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        idx = int(probs.argmax().item())
        return self.classes_map[str(idx)], float(probs[idx])

