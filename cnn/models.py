''' Where the CNN models are defined. '''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    '''
    A CNN with 1 convolutional layer and 1 linear layer.
    '''
    def __init__(self, num_channels):
        super().__init__()
                
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=num_channels, 
                      kernel_size=(3, 3),
                      stride=1,),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Linear(num_channels * 63 * 39, 2)

        
    def forward(self, x):
        x = self.layer1(x)
        x = torch.flatten(x, 1)
        x = F.softmax(self.layer2(x), dim=1)
        return x