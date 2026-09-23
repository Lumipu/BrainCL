# coding=utf-8


import argparse
import configparser
import os
import random
import numpy as np
import torch
import torch.backends.cudnn
from conf.loadpath import PathInfo
from dataloader.mydataloader import MyDataLoader
from model.brainclmodel import BrainCLModel


class BrainNetworkAnalysis(object):
    """Brain network analysis."""
    cfg_file_path = './conf/config.ini'

    @classmethod
    def main(cls):
        args = cls.__arg_parse()
        cls.__setup(args.seed, args.cuda)
        path = cls.__get_path()
        repeated_cv_dataloader = MyDataLoader(dataset_path=path.dataset_path,n_repeats=args.n_repeats,
                                              n_splits=args.n_splits, seed=args.seed, batch_size=args.batch_size)
        model = BrainCLModel(args.lr, args.weight_decay, args.max_epochs, repeated_cv_dataloader)
        model.train()

    @classmethod
    def __get_path(cls):
        """Load path settings from the configuration file."""
        parser = configparser.ConfigParser()
        parser.read(cls.cfg_file_path, encoding='utf-8')
        path = PathInfo(parser)
        return path

    @staticmethod
    def __arg_parse():
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog='Arguments')
        parser.add_argument('--cuda', type=str, default='1', help='cuda device')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        parser.add_argument('--seed', type=int, default=63, help='random seed')
        parser.add_argument('--n_splits', type=int, default=10, help='n-fold')
        parser.add_argument('--n_repeats', type=int, default=10, help='n_repeats')
        parser.add_argument('--lr', type=float, default=1e-4, help='leaning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
        parser.add_argument('--max_epochs', type=int, default=200, help='max epochs')
        args = parser.parse_args()
        return args

    @staticmethod
    def __setup(seed, cuda):
        """Set random seeds and configure CUDA."""
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    BrainNetworkAnalysis.main()
