import argparse
import glob
import os
import random
import pandas as pd
import numpy as np
import shutil
from utils import get_module_logger
import glob


def split(source, train_prop, test_prop, valid_prop):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        #- destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # TODO: Implement function
    #source = './data/processed'

    data = glob.glob(source+'/*')
    random.shuffle(data)  
    tab = pd.DataFrame(columns=['paths'],data=data)
    tab['sample'] = 0
    training = list(tab.sample(frac=float(train_prop))['paths'])
    tab['sample'][tab['paths'].isin(training)] = 'train'
    tt = int(len(data)*float(test_prop))
    testing = list(tab[tab['sample']==0].sample(n=tt)['paths'])
    tab['sample'][tab['paths'].isin(testing)] = 'test'
   
    vv = int(len(data)*float(valid_prop))
    validation = list(tab[tab['sample']==0].sample(n=vv)['paths'])

    if not os.path.isdir('./training/'):
        os.mkdir('./training/')
    for i in training:
        shutil.move(i,  './training/')

    if not os.path.isdir('./testing/'):
        os.mkdir('./testing/')
    for i in testing:
        shutil.move(i, './testing/')
    
    if not os.path.isdir('./validation/'):
        os.mkdir('./validation/')
    for i in validation:
        shutil.move(i,  './validation/')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--train_prop', required=True,
                        help='training proportion')
    parser.add_argument('--test_prop', required=True,
                        help='testing proportion')
    parser.add_argument('--valid_prop', required=True,
                        help='validation proportion')

    #parser.add_argument('--destination', required=True,
    #                    help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.train_prop, args.test_prop, args.valid_prop) #, args.destination

