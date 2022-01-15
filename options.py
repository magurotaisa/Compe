""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse

# pylint: disable=C0103,C0301,R0903,W0622


class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument(
            '--train_dataset', default="./input_data/train/train.csv", help='train data dir')
        self.parser.add_argument(
            '--test_dataset', default="./input_data/test/test_easy.csv", help='test data dir')
        self.parser.add_argument(
            '--save', default="./test_easy.csv", help='test data dir')
        self.parser.add_argument(
            '--weight', default="./version1_result/easy/checkpoints/sample-max-epoch=28-val_loss=0.017.ckpt", help='test data dir')

        # self.parser.add_argument(
        #     '--val_data_normal', default='../cow_data/cow_clip_2/test/0.normal', help='val normal data dir')
        # self.parser.add_argument(
        #     '--val_data_abnormal', default='../cow_data/cow_clip_2/test/1.abnormal', help='val abnormal data dir')

        self.parser.add_argument(
            '--batch', type=int, default=16, help='batch size')
        self.parser.add_argument(
            '--num_workers', type=int, default=8, help='num_workes')
        self.parser.add_argument(
            '--epoch', type=int, default=30, help='num epoch')
        self.parser.add_argument(
            '--yaml', type=str, default="./lightning_logs/version_1/hparams.yaml", help='test yaml path')
        self.parser.add_argument(
            '--skip', type=int, default=45, help='test yaml path')

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()

        return self.opt
