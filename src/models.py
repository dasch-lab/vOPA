"""
Baseline: pre-trained Densenet 161 with adapted head and custom final layer
"""

import torch
import torch.nn as nn


class Dense161_model(nn.Module):
    def __init__(self):
        super().__init__()

        back_bone = torch.hub.load("pytorch/vision:v0.10.0", "densenet161", pretrained=True)
        input_layer = nn.Conv2d(4, 3, 1)
        self.feature_extractor = nn.Sequential(
            input_layer, back_bone.features, nn.AvgPool2d(16), nn.Flatten()
        )
        
        self.head = nn.Linear(2208, 2)
        

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.head(x)
        return x

class VGG16_model(nn.Module):
    def __init__(self):
        super().__init__()

        back_bone = torch.hub.load("pytorch/vision:v0.10.0", "vgg16", pretrained=True)
        input_layer = nn.Conv2d(4,3,1)
        self.feature_extractor = nn.Sequential(
            input_layer, back_bone.features, back_bone.avgpool, nn.Flatten()
        )

        self.head = nn.Sequential(nn.Linear(25088, 5000), nn.Linear(5000, 2))
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.head(x)
        return x

class ResNet18_model(nn.Module):
    def __init__(self):
        super().__init__()

        back_bone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        # back_bone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        input_layer = nn.Conv2d(4,3,1)
        self.feature_extractor = nn.Sequential(input_layer, nn.Sequential(*(list(back_bone.children())[:-1])), nn.Flatten())
        self.head = nn.Linear(512,2)
    
    def forward(self,x):
        x = self.feature_extractor(x)
        x = self.head(x)
        return x


