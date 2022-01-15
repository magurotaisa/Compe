from pytorch_lightning import LightningDataModule
from data import Mydataset, Mydataset_test
from torch.utils.data import DataLoader
import torch


class MydataModule(LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.train_data = opt.train_dataset
        self.test_data = opt.test_dataset

        self.batch = opt.batch
        self.num_workers = opt.num_workers

        mydataset = Mydataset(self.train_data, skip=self.opt.skip)
        self.all_dataset = mydataset

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            n_samples = len(self.all_dataset)
            train_size = int(len(self.all_dataset) * 0.8)
            val_size = n_samples - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.all_dataset, [train_size, val_size])
        elif stage == "test":
            self.test_dataset = Mydataset_test(
                self.test_data, skip=self.opt.skip, sc=self.get_sc())

    def get_sc(self):
        return self.all_dataset.get_sc()

    def get_colnum(self):
        return self.all_dataset.get_colnum()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.opt.batch, num_workers=self.opt.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.opt.batch, num_workers=self.opt.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.opt.batch, num_workers=self.opt.num_workers)
