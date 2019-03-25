import math

import cv2
import numpy
import os
from skimage.measure import compare_ssim as ssim

from utils.options import args


def psnr(img1, img2):
    mse = numpy.mean(numpy.square((img1 - img2)))
    if mse == 0:
        return 100
    return 10 * math.log10(255 ** 2 / mse)


def scoring():
    reals = os.listdir('../data/div2k_valid/hr/')
    path = './logs/' + args.version + '/eval/'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log = open(path + str(args.eval_epoch) + '.txt', 'a+')
    generated = os.listdir('./out/' + args.version + '/eval/epoch' + str(args.eval_epoch))
    reals = sorted(reals, key=lambda x: int(os.path.splitext(x)[0]))
    generated = sorted(generated, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    psnr_sum = 0
    ssim_sum = 0
    for r, g in zip(reals, generated):
        r = cv2.imread('../data/div2k_valid/hr/' + r)
        g = cv2.imread('./out/' + args.version + '/eval/epoch' + str(args.eval_epoch) + '/' + g)

        psnr_ = psnr(r, g)
        ssim_ = ssim(r, g, multichannel=True, sigma=1.5, use_sample_covariance=False, gaussian_weights=True)
        psnr_sum += psnr_
        ssim_sum += ssim_
        log.write('PNSR:{}\tSSIM:{}'.format(psnr_, ssim_))
    log.write('AVG_PNSR:{}\tAVG_SSIM:{}\n'.format(psnr_sum / len(generated), ssim_sum / len(generated)))


if __name__ == '__main__':
    args.version = 'v3'
    args.eval_epoch = 35
    scoring()
