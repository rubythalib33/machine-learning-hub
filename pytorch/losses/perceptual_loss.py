import torch.nn as nn
import torch
from torchvision import models


class PerceptualLoss(nn.Module):
    def __init__(self, backbone:nn.Module=models.vgg19(weights='IMAGENET1K_V1').features, device:str="cpu"):
        super(PerceptualLoss, self).__init__()
        self.backbone = backbone.eval().to(device)
        self.loss = nn.MSELoss()

        for param in self.backbone.parameters():
            param.requires_grad = False
    

    def forward(self, predict:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        predict_features = self.backbone(predict)
        target_features = self.backbone(target)

        return self.loss(predict_features,target_features)