from preproc_utils import train_test_valid_split

import pandas as pd
import argparse
from os.path import join


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splits csv file on train/valid/test csv files')
    
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Path to csv file')
    parser.add_argument('-v', '--valid_frac', type=float, default=0.2, required=False,
                        help='Fraction of data taken for validation set. Default: 0.2')
    parser.add_argument('-t', '--test_frac', type=float, default=0., required=False,
                        help='Fraction of data taken for test set. Default: 0.')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to save splitted data')
    
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    train, valid, test = train_test_valid_split(data, args.valid_frac, args.test_frac)
    train.to_csv(join(args.path, 'train.csv'), index=False)
    valid.to_csv(join(args.path, 'valid.csv'), index=False)
    test.to_csv(join(args.path, 'test.csv'), index=False)
