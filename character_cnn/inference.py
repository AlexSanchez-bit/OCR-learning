from torch.cuda import is_available
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
from ml_lib.visualizations.feature_map_inspector import FeatureMapInspector
from ml_lib.visualizations.grad_cam import grad_cam




class ModelInference():
    def __init__(self):
        self.classes_map = load_from_json()
        checkpoint_manager = BaseCheckpointLoader(MODEL_STATE_PATH)
        last_state = checkpoint_manager.load_checkpoint_data()
        self.model = CharacterModel()
        self.feature_inspector = FeatureMapInspector()
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

    def inference_with_heatmap(self,img):
        image_tensor = self.preprocess(img).unsqueeze(0)
        if torch.cuda.is_available():
            image_tensor=image_tensor.to('cuda')
            self.model.to('cuda')
        image_tensor.requires_grad_(True)
        self.model.eval()
        #get the output
        def handler(x):
            """ forces gradient retention """
            x.retain_grad()
            return x

        self.feature_inspector.watch_live('conv2',self.model.conv2,handler)
        output=self.model(image_tensor)
        # get a disconnected output to predictions
        detached_out = output.detach().cpu()
        probs = torch.softmax(detached_out, dim=1)[0]
        idx = int(probs.argmax().item())
        # reset gradients to calculated them based on the activated logit
        self.model.zero_grad()
        # find max activated logit, and its score value
        max_logit = output.argmax(dim=1)
        score = output[0,max_logit[0]]
        # calculate the gradients of this result given the image
        score.backward()
        result = grad_cam(self.feature_inspector.get_data('conv2'),img)
        result = result.unsqueeze(dim=0).unsqueeze(dim=0)
        interpolated = F.interpolate(result,size=image_tensor.squeeze(dim=0).squeeze(dim=0).shape,mode='bilinear', align_corners=False)
        #got absolute value of the results 1) representation 2) we are interested only on the importance of each pixel
        saliency= image_tensor.grad.detach().cpu().abs()

        fig, axis = plt.subplots(1,2)
        #saliency map
        axis[0].set_title("saliency map")
        axis[0].imshow(
            img,
            cmap="gray"
        )
        axis[0].imshow(
            saliency[0, 0]/saliency[0,0].max().item(),
            cmap="jet",
            alpha=0.5
        )
        #gradcam
        axis[1].set_title("grad cam")
        axis[1].imshow(
            img,
            cmap="gray"
        )
        axis[1].imshow(
            interpolated.squeeze(dim=0).squeeze(dim=0),
            cmap="jet",
            alpha=0.4
        )
        return self.classes_map[str(idx)], float(probs[idx]),fig,axis


    def inference(self,img):
        image_tensor = self.preprocess(img).unsqueeze(0)
        if torch.cuda.is_available():
            image_tensor=image_tensor.to('cuda')
            self.model.to('cuda')
        with torch.no_grad():
            probs = torch.softmax(self.model(image_tensor), dim=1)[0]
        idx = int(probs.argmax().item())
        return self.classes_map[str(idx)], float(probs[idx])

