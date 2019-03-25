import torch.optim as optim
from torch import nn
import torch
from Loss.Discriminator import Discriminator
from utils import log
from options import args


class GAN(nn.Module):
    def __init__(self, k=5, patch=256, out=1024, depth=5, cond=False,nf=64):
        super(GAN, self).__init__()
        self.loss = 0
        self.k = k
        self.nf = nf
        self.d = Discriminator(patch_size=patch, depth=depth)
        trainable = filter(lambda x: x.requires_grad, self.d.parameters())
        self.optim = optim.Adam(trainable, lr=args.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.cond = cond
        self.out = out
        self.depth = depth
        self.classif = self.classifier(patch)
        if cond:
            self.class_real = self.classifier(patch // 2)

    def classifier(self, patch_size):

        linear_input = ((patch_size // 2 ** (self.depth // 2 + 1)) ** 2) * (self.nf * (2 ** (self.depth // 2)))
        classifier = [nn.Linear(linear_input, self.out),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                      nn.Linear(self.out, 1)]
        return nn.Sequential(*classifier)

    def forward(self, fake, real):
        self.loss = 0
        fake_noback = fake.detach()

        for i in range(self.k):
            self.optim.zero_grad()

            d_fake = self.d(fake_noback)
            d_fake = d_fake.view(d_fake.size(0), -1)
            d_fake = self.classif(d_fake)
            d_real = self.d(real)
            d_real = d_real.view(d_real.size(0), -1)
            d_real = self.classif(d_real)


            ##TODO chnage real to 0.9
            ##TODO print each BCE to check how D is doing
            # https://github.com/tensorlayer/srgan/blob/34ebf0ff0ca788980ec818daa6614b626051e389/main.py#L65
            label_real = torch.ones_like(d_real) - 0.1
            label_fake = torch.zeros_like(d_fake)
            bce_real = self.criterion(d_real, label_real)
            bce_fake = self.criterion(d_fake, label_fake)
            loss_d = bce_real + bce_fake

            log("bce real:{:+.3f}\tbce fake:{:+.3f}".format(bce_real, bce_fake),
                "./logs/" + args.version + "/discrminiator_bce_.txt")

            self.optim.zero_grad()
            self.loss += loss_d.item()
            loss_d.backward()
            self.optim.step()

        d_fake_back = self.d(fake)
        d_fake_back = d_fake_back.view(d_fake_back.size(0), -1)
        d_fake_back = self.classif(d_fake_back)
        label_real = torch.ones_like(d_fake_back)
        loss_g = self.criterion(d_fake_back, label_real)

        return loss_g
