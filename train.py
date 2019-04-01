import os
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from Loss import Loss
from utils.dataset import DatasetManager, ImageManager
from utils.options import args
from utils.utils import load


def train():
    from model.EDSR import EDSR
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    transform_ = transforms.Compose([
        transforms.ToTensor()
    ])
    data = DatasetManager(dataset='DIV2K', transform=transform_, psize=[args.patch_size, args.patch_size])
    dataMan = DataLoader(data, batch_size=1)
    criterion = Loss.Loss(device, patch_size=args.patch_size * 2, losses=args.loss, k=args.k, out=args.o,
                          depth=args.depth)

    EDSR = EDSR(args.scale, args.res_length)

    EDSR.to(device)
    trainable = filter(lambda x: x.requires_grad, EDSR.parameters())
    optimizer = optim.Adam(trainable, lr=args.lr,weight_decay=200)
    EDSR, criterion, optimizer, epoch, runningavg = load(EDSR, criterion, optimizer, load=args.load,path='./checkpoints/' + args.version)

    for epoch in range(epoch, args.epochs):

        running_loss = 0.0
        sep_losses = [0 for _ in range(len(args.loss.split(',')))]

        for i, (X, y) in enumerate(dataMan, 0):

            patchesX = data.extract_patches(X, batch_first=True)
            patchesY = data.extract_patches(y, size=[data.psize[0] * 2, data.psize[1] * 2], batch_first=True)

            if patchesX.size(1) == patchesY.size(1):
                patchManager = DataLoader(ImageManager(patchesX[0], patchesY[0]), batch_size=args.batch)

                for j, (px, py) in enumerate(patchManager, 0):

                    optimizer.zero_grad()
                    outputs = EDSR(px.to(device))
                    loss = criterion.forward(outputs.to(device), py.to(device))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    sep_losses = [sepl_new.item() + sepl_old for sepl_new, sepl_old in
                                  zip(criterion.losses, sep_losses)]
                sep_losses = [sepl / patchesY.size(1) for sepl in sep_losses]
                running_loss /= patchesY.size(1)
                runningavg['LOSS'].append(running_loss)
                for idx, loss in enumerate(args.loss.split(','),0):
                    runningavg[loss].append(sep_losses[idx])
                log(log_type(running_loss, sep_losses, runningavg, epoch, i),
                    './logs/' + args.version + '/loss_logs.txt')
                if i % 20 == 0:
                    os.makedirs(os.path.dirname('./out/'+args.version+'/'), exist_ok=True)
                    torchvision.utils.save_image(outputs[-1],
                                                 './out/'+args.version+'/{}_{}_pred.png'.format(epoch + 1, i + 1))
                    torchvision.utils.save_image(patchesY[0][-1],
                                                 './out/'+args.version+'/{}_{}_hr.png'.format(epoch + 1, i + 1))
                    torchvision.utils.save_image(patchesX[0][-1],
                                                 './out/'+args.version+'/{}_{}_lr.png'.format(epoch + 1, i + 1))

                    sep_losses = [0, 0, 0]
                    running_loss = 0.0

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': EDSR.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': criterion.state_dict(),
            'ema': runningavg
        }, filename='./checkpoints/' + args.version + '/Epoch_' + args.version + '_{}.pt'.format(epoch + 1))
