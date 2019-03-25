import train
from options import args
import eval

if __name__ == '__main__':
    if not args.eval:
        train.train()
    else:
        eval.eval()

