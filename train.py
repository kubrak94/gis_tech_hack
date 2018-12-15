from train_helpers import run_train

import argparse

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--hparams', type=str, required=True)
    parser.add_argument('--logdir', type=str, required=True)
    
    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)'
    )
    
    args = parser.parse_args()

    run_train(args)
