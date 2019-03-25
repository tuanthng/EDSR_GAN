import torch.nn as nn



class Discriminator(nn.Module):
    def __init__(self, ni=3, nf=64, depth=5, patch_size=256):
        super(Discriminator, self).__init__()
        self.ni = ni
        self.nf = nf
        self.depth = depth
        self.patch_size = patch_size
        self.features = self.convBlock(self.ni,self.nf)
        #self.output = self.classifier()


    def conv(self, ni, nf, kernel_size=3, stride=1):
        layers = [nn.Conv2d(ni, nf, kernel_size, padding=1, stride=stride), nn.BatchNorm2d(nf), nn.LeakyReLU(negative_slope=0.2,inplace=True)]
        return nn.Sequential(*layers)

    def convBlock(self, ni, nf):
        features = [self.conv(ni, nf)]

        ##https://github.com/tensorlayer/srgan/blob/34ebf0ff0ca788980ec818daa6614b626051e389/model.py#L105
        for i in range(self.depth):
            ni = nf
            if i % 2 == 1:
                stride = 1
                nf = 2 * nf
            else:
                stride = 2
            features.append(self.conv(ni, nf, stride=stride))
        return nn.Sequential(*features)


    # def classifier(self):
    #
    #     linear_input = ((self.patch_size//2**(self.depth//2+1))**2)*(self.nf*(2**(self.depth//2)))
    #     classifier = [nn.Linear(linear_input, self.out),
    #                  nn.LeakyReLU(negative_slope=0.2,inplace=True),
    #                  nn.Linear(self.out, 1)]
    #     return nn.Sequential(*classifier)

    def forward(self, x):
        features = self.features(x)
        return features
