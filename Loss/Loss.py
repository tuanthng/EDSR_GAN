from torch import nn
from .GAN import GAN
from .VGG import VGG


class Loss(nn.modules.loss._Loss):

    def __init__(self, device, patch_size=256, losses='GAN,VGG,MSE', k=1, out=1024, depth=5):
        super(Loss, self).__init__()
        self.loss_module = nn.ModuleList()
        self.loss = losses
        if losses.find('GAN') != -1:
            self.gan = GAN(patch=patch_size, k=k, out=out, depth=depth)
            self.loss_module.append(self.gan)
        if losses.find('VGG') != -1:
            self.vgg = VGG()
            self.loss_module.append(self.vgg)
        if losses.find('MSE') != -1:
            self.mse = nn.MSELoss()
            self.loss_module.append(self.mse)

        self.loss_module.to(device)
        self.losses = []

    def forward(self, sr, hr):
        # https://github.com/tensorlayer/srgan/blob/34ebf0ff0ca788980ec818daa6614b626051e389/main.py#L89
        losses = []
        if self.loss.find('VGG') != -1:
            vggloss = self.vgg(sr, hr)
            losses.append(5 * vggloss)

        if self.loss.find('GAN') != -1:
            ganloss = self.gan(sr, hr)
            losses.append(ganloss)

        if self.loss.find('MSE') != -1:
            self.mse.zero_grad()
            mse = self.mse(sr, hr)
            losses.append(6 * mse)


        self.losses = losses

        return sum(losses)
