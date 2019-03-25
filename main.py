import train
from utils.options import args
from utils import eval

if __name__ == '__main__':
    if not args.eval:
        train.train()
    else:
        eval.eval()

