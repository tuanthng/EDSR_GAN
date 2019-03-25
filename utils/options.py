import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p","--patch_size", help="increase output verbosity",default=48,
                    type=int)
parser.add_argument('-s','--scale',help='upscale factor',default=2,
                    type=int)
parser.add_argument('-l','--loss',help='type of loss to use',default='VGG,GAN,MSE',
                    type=str,choices=['GAN','MSE','VGG','GAN,MSE','VGG,GAN','VGG,MSE'])
parser.add_argument('-e','--epochs',help='how many epochs to train',default=300,
                    type=int)
parser.add_argument('-v','--version',help='version of the model for the logs',default='v1',
                    type=str)
parser.add_argument('--load',help='load a previous model to continue training',default=False,
                    action='store_true')
parser.add_argument('-r','--res_length',help='Length of the residual block',default=32,
                    type=int)
parser.add_argument('--lr',help='Learning rate',default=1e-4,
                    type=int)
parser.add_argument('--cuda',help='GPU to use',default='cuda:1',
                    type=str)

parser.add_argument('-k',help='how many times should we train D over a batch',default=1,
                    type=int)
parser.add_argument('-o', help="Size of the classifier",default=1024,
                    type=int)
parser.add_argument('--depth',help='depth of the conv net',default=5,
                    type=int)
parser.add_argument('--batch',help='batch size',default=100,
                    type=int)
parser.add_argument('--eval_epoch',help='Which epoch to evaluate for the current version',default=1,
                    type=int)
parser.add_argument('--eval',help='Which epoch to evaluate for the current version',default=False,
                    action='store_true')

args = parser.parse_args()
