from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch

from options import args


def log_type(running_loss, sep_losses, runningavg, epoch, i):
    if args.loss == 'VGG,GAN,MSE':
        return '[{}, {}]\tloss:{:+.3f} {:+.3f}\t VGG:{:+.3f} {:+.3f}\t GAN:{:+.3f} {:+.3f}\t MSE:{:+.3f} {:+.3f}'.format(
            epoch + 1, i,
            running_loss, sum(runningavg['LOSS'][-50:]) / len(runningavg['LOSS'][-50:]),
            sep_losses[0], sum(runningavg['VGG'][-50:]) / len(runningavg['LOSS'][-50:]),
            sep_losses[1], sum(runningavg['GAN'][-50:]) / len(runningavg['LOSS'][-50:]),
            sep_losses[2], sum(runningavg['MSE'][-50:]) / len(runningavg['LOSS'][-50:]))

    elif args.loss == 'GAN' or args.loss == 'MSE' or args.loss == 'VGG':
        return '[{}, {}]\t{}:{:+.3f} {:+.3f}'.format(
            epoch + 1, i, args.loss,
            running_loss, sum(runningavg['LOSS'][-50:]) / len(runningavg['LOSS'][-50:]))

    elif args.loss == 'GAN,MSE' or args.loss == 'VGG,GAN' or args.loss == 'VGG,MSE':
        first, second = args.loss.split(',')
        return '[{}, {}]\tloss:{:+.3f} {:+.3f}\t {}:{:+.3f} {:+.3f}\t {}:{:+.3f} {:+.3f}'.format(
            epoch + 1, i, first, second,
            running_loss, sum(runningavg['LOSS'][-50:]) / len(runningavg['LOSS'][-50:]),
            sep_losses[0], sum(runningavg['VGG'][-50:]) / len(runningavg['LOSS'][-50:]),
            sep_losses[1], sum(runningavg['GAN'][-50:]) / len(runningavg['LOSS'][-50:]))


def log(text, path='./logs/logs_patch48_v6.txt'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a+")
    f.write(text + '\n')


def plot(file):
    res = {}
    f = open(file, 'r')
    old_epoch = 0
    loss = '\s+\w+\:([-+]?\d+\.\d+)\s([-+]?\d+\.\d+)'
    for line in f.readlines():

        rx = re.compile(
            '\[(\d+),\s(\d+)\]'+loss*4,
            re.VERBOSE)
        epoch, photo, loss, mloss, vgg, mvgg, gan, mgan, mse, mmse = rx.match(line).group(1), \
                                                                     rx.match(line).group(2), \
                                                                     rx.match(line).group(3), \
                                                                     rx.match(line).group(4), \
                                                                     rx.match(line).group(5), \
                                                                     rx.match(line).group(6), \
                                                                     rx.match(line).group(7), \
                                                                     rx.match(line).group(8), \
                                                                     rx.match(line).group(9), \
                                                                     rx.match(line).group(10)
        if old_epoch != int(epoch):
            res[epoch] = {}
            res[epoch]['LOSS'] = []
            res[epoch]['mloss'] = []
            res[epoch]['VGG'] = []
            res[epoch]['mvgg'] = []
            res[epoch]['GAN'] = []
            res[epoch]['mgan'] = []
            res[epoch]['MSE'] = []
            res[epoch]['mmse'] = []
            res[epoch]['LOSS'].append(float(loss))
            res[epoch]['mloss'].append(float(mloss))

            res[epoch]['VGG'].append(float(vgg))
            res[epoch]['mvgg'].append(float(mvgg))

            res[epoch]['GAN'].append(float(gan))
            res[epoch]['mgan'].append(float(mgan))

            res[epoch]['MSE'].append(float(mse))
            res[epoch]['mmse'].append(float(mmse))
            old_epoch = int(epoch)
        else:
            res[epoch]['LOSS'].append(float(loss))
            res[epoch]['mloss'].append(float(mloss))

            res[epoch]['VGG'].append(float(vgg))
            res[epoch]['mvgg'].append(float(mvgg))

            res[epoch]['GAN'].append(float(gan))
            res[epoch]['mgan'].append(float(mgan))

            res[epoch]['MSE'].append(float(mse))
            res[epoch]['mmse'].append(float(mmse))

    return res


def extract_number(f):
    s = re.findall('\d+$', f)
    return int(s[0]) if s else -1, f


def load(model, loss, optim, load=True, path='./checkpoints/'):
    if load:
        files = os.listdir(path)
        checkpoint = torch.load('{}{}'.format(path, max(files, key=extract_number)))

        model.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        loss.load_state_dict(checkpoint['LOSS'])
        return model, loss, optim, checkpoint['epoch'], checkpoint['ema']
    else:
        runningavg = {'LOSS': [], 'VGG': [], 'GAN': [], 'MSE': []}
        return model, loss, optim, 0, runningavg


def save_checkpoint(state, filename='checkpoint.pt'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


if __name__ == '__main__':
    xepoch = 0
    res = plot('logs/randomcrop/loss_logs.txt')
    i = 1

    for key in sorted(res.keys(), key=lambda x: int(x)):
        size = len(res[key]['LOSS'])
        y = np.arange(xepoch, size + xepoch)
        plt.plot(y, res[key]['LOSS'], color='b')
        plt.plot(y, res[key]['mloss'], color='r', label="EmA Loss")

        # plt.plot(y, res[key]['VGG'], color='g')
        plt.plot(y, res[key]['mvgg'], color='b', label="EmA VGG")
        plt.plot(y, res[key]['mgan'], color='g', label="EmA GAN")
        plt.plot(y, res[key]['mmse'], color='y', label="EmA MSE")
        # plt.axvline(x=xepoch + size, color='k', linestyle="--")
        i += 1
        xepoch += size
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig('test.png')
