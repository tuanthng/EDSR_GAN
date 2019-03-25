import os
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset


class DatasetManager(Dataset):

    def __init__(self, dataset='DIV2K', loader=None, transform=None, psize=None, pstride=None):

        if pstride is None:
            self.pstride = [0.6, 0.6]
        else:
            self.pstride = pstride

        if psize is None:
            self.psize = [128,128]
        else:
            self.psize = psize
        self.dataset = dataset
        self.root = ''
        self.X = None
        self.y = None
        self.transform = transform
        if self.dataset == 'DIV2K':
            self.X_dir = 'lr_2x/'
            self.y_dir = 'hr/'
            self.root = '../data/div2k/'
            self.X = os.listdir(self.root + self.X_dir)
            self.y = os.listdir(self.root + self.y_dir)
            self.X.sort()
            self.y.sort()
        if self.dataset == 'DIV2K_val':
            self.X_dir = 'lr_2x/'
            self.y_dir = 'hr/'
            self.root = '../data/div2k_valid/'
            self.X = os.listdir(self.root + self.X_dir)
            self.y = os.listdir(self.root + self.y_dir)
            self.X.sort()
            self.y.sort()
        if loader is None:
            self.loader = Image.open

    def extract_patches(self, img, size=None, stride=None, batch_first=False):
        #https://gist.github.com/dem123456789/23f18fd78ac8da9615c347905e64fc78
        if size is None:
            size = self.psize
        else:
            size = size
        if stride is None:
            stride = self.pstride
        else:
            size = size
        patch_H, patch_W = size[0], size[1]
        if img.size(2) < patch_H:
            num_padded_H_Top = (patch_H - img.size(2)) // 2
            num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
            padding_H = nn.ConstantPad2d((0, 0, num_padded_H_Top, num_padded_H_Bottom), 0)
            img = padding_H(img)
        if img.size(3) < patch_W:
            num_padded_W_Left = (patch_W - img.size(3)) // 2
            num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
            padding_W = nn.ConstantPad2d((num_padded_W_Left, num_padded_W_Right, 0, 0), 0)
            img = padding_W(img)
        step_int = [0, 0]
        step_int[0] = int(patch_H * stride[0]) if (isinstance(stride[0], float)) else stride[0]
        step_int[1] = int(patch_W * stride[1]) if (isinstance(stride[1], float)) else stride[1]
        patches_fold_H = img.unfold(2, patch_H, step_int[0])
        if (img.size(2) - patch_H) % step_int[0] != 0:
            patches_fold_H = torch.cat((patches_fold_H, img[:, :, -patch_H:, ].permute(0, 1, 3, 2).unsqueeze(2)), dim=2)
        patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])
        if (img.size(3) - patch_W) % step_int[1] != 0:
            patches_fold_HW = torch.cat(
                (patches_fold_HW, patches_fold_H[:, :, :, -patch_W:, :].permute(0, 1, 2, 4, 3).unsqueeze(3)), dim=3)
        patches = patches_fold_HW.permute(2, 3, 0, 1, 4, 5)
        patches = patches.reshape(-1, img.size(0), img.size(1), patch_H, patch_W)
        if batch_first:
            patches = patches.permute(1, 0, 2, 3, 4)
        return patches

    def reconstruct_from_patches(self, patches, img_shape, stride=None, batch_first=False):
        ##TODO check padding while reconstructing
        if stride is None:
            stride = self.pstride
        if batch_first:
            patches = patches.permute(1, 0, 2, 3, 4)
        patch_H, patch_W = patches.size(3), patches.size(4)
        img_size = (patches.size(1), patches.size(2), max(img_shape[0], patch_H), max(img_shape[1], patch_W))

        step_int = [0, 0]
        step_int[0] = int(patch_H * stride[0]) if (isinstance(stride[0], float)) else stride[0]
        step_int[1] = int(patch_W * stride[1]) if (isinstance(stride[1], float)) else stride[1]
        nrow, ncol = 1 + (img_size[-2] - patch_H) // step_int[0], 1 + (img_size[-1] - patch_H) // step_int[1]
        r_nrow = nrow + 1 if ((img_size[2] - patch_H) % step_int[0] != 0) else nrow
        r_ncol = ncol + 1 if ((img_size[3] - patch_W) % step_int[1] != 0) else ncol
        patches = patches.reshape(r_nrow, r_ncol, img_size[0], img_size[1], patch_H, patch_W)
        img = torch.zeros(img_size, device=patches.device)
        overlap_counter = torch.zeros(img_size, device=patches.device)
        for i in range(nrow):
            for j in range(ncol):
                img[:, :, i * step_int[0]:i * step_int[0] + patch_H, j * step_int[1]:j * step_int[1] + patch_W] += \
                    patches[i, j,]
                overlap_counter[:, :, i * step_int[0]:i * step_int[0] + patch_H,
                j * step_int[1]:j * step_int[1] + patch_W] += 1
        if (img_size[2] - patch_H) % step_int[0] != 0:
            for j in range(ncol):
                img[:, :, -patch_H:, j * step_int[1]:j * step_int[1] + patch_W] += patches[-1, j,]
                overlap_counter[:, :, -patch_H:, j * step_int[1]:j * step_int[1] + patch_W] += 1
        if (img_size[3] - patch_W) % step_int[1] != 0:
            for i in range(nrow):
                img[:, :, i * step_int[0]:i * step_int[0] + patch_H, -patch_W:] += patches[i, -1,]
                overlap_counter[:, :, i * step_int[0]:i * step_int[0] + patch_H, -patch_W:] += 1
        if (img_size[2] - patch_H) % step_int[0] != 0 and (img_size[3] - patch_W) % step_int[1] != 0:
            img[:, :, -patch_H:, -patch_W:] += patches[-1, -1,]
            overlap_counter[:, :, -patch_H:, -patch_W:] += 1
        img /= overlap_counter
        if img_shape[0] < patch_H:
            num_padded_H_Top = (patch_H - img_shape[0]) // 2
            num_padded_H_Bottom = patch_H - img_shape[0] - num_padded_H_Top
            img = img[:, :, num_padded_H_Top:-num_padded_H_Bottom, ]
        if img_shape[1] < patch_W:
            num_padded_W_Left = (patch_W - img_shape[1]) // 2
            num_padded_W_Right = patch_W - img_shape[1] - num_padded_W_Left
            img = img[:, :, :, num_padded_W_Left:-num_padded_W_Right]
        return img

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.transform(self.loader(os.path.join(self.root + self.X_dir + self.X[index]))), \
               self.transform(self.loader(os.path.join(self.root + self.y_dir + self.y[index])))


class ImageManager(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.size(0)
