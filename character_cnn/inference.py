from torchvision import transforms
import torch

from character_cnn.constants import MODEL_STATE_PATH
from character_cnn.indexation.save_indexed_chars import load_from_json
from character_cnn.models.character_model import CharacterModel
from ml_lib.checkpoint.checkpoint_loader import BaseCheckpointLoader



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

    def inference(self,img):
        x = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        idx = int(probs.argmax().item())
        return self.classes_map[str(idx)], float(probs[idx])

