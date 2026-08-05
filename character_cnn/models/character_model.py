import torch.nn as nn
from torch import flatten
from ml_lib.model_data_extractor.convolution_presenter import ConvolutionPresenter


class CharacterModel(nn.Module,ConvolutionPresenter):
    def __init__(self) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=0.2) 
        self.conv1 = nn.Conv2d(
            1,
            32,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2)

        self.fc1= nn.Linear(64*7*7,265)

        self.fc2= nn.Linear(265,47)


    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.relu(x)
        x=flatten(x,start_dim=1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x

    def get_convolutions(self)->List[torch.Tensor]:
      return [self.conv1,self.conv2]

