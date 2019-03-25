import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


from dataset import DatasetManager
from options import args


def eval():
    os.makedirs(os.path.dirname('./out/' + args.version + '/eval/epoch' + str(args.eval_epoch)+'/'),
                exist_ok=True)
    from EDSR import EDSR
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    transform_ = transforms.Compose([
        transforms.ToTensor()
    ])
    data = DatasetManager(dataset='DIV2K_val', transform=transform_, psize=[48, 48])
    dataMan = DataLoader(data, batch_size=1)

    EDSR = EDSR(args.scale, args.res_length)
    checkpoint = torch.load('./checkpoints/' + args.version + '/Epoch_' + args.version + '_' + str(args.eval_epoch) + '.pt',
                            map_location=lambda storage, loc: storage)
    EDSR.load_state_dict(checkpoint['state_dict'])
    # EDSR.to(device)
    EDSR.eval()
    epoch = checkpoint['epoch']
    del checkpoint

    for i, (X, y) in enumerate(dataMan, 0):
        outputs = EDSR(X)
        # patchesX = data.extract_patches(X, batch_first=True)
        # patchesY = data.extract_patches(y, size=[data.psize[0] * 2, data.psize[1] * 2], batch_first=True)
        # patches = torch.empty([]).cpu()
        # if patchesX.size(1) == patchesY.size(1):
        #     patchManager = DataLoader(ImageManager(patchesX[0], patchesY[0]), batch_size=10)
        #     for j, (px, py) in enumerate(patchManager, 0):
        #         if j == 0:
        #             outputs = EDSR(px)
        #             patches = outputs
        #         else:
        #             outputs = EDSR(px)
        #             patches = torch.cat((patches, outputs))
        #
        #     patches = patches.unsqueeze(0)
        #     image = data.reconstruct_from_patches(patches, [y.size(-2), y.size(-1)], batch_first=True)
        torchvision.utils.save_image(outputs[-1],
                                     './out/' + args.version + '/eval/epoch' + str(args.eval_epoch) +
                                     '/eval_{}_{}.png'.format(
                                         epoch, i + 1))

