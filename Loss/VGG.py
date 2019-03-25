import torch
import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        self.vgg = nn.Sequential(*modules[:35])
        self.mse = nn.MSELoss()

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        self.mse.zero_grad()
        loss = self.mse(vgg_sr, vgg_hr)

        return loss
