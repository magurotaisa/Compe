from torch.utils.data import Dataset
# import numpy as np
import pandas as pd
# from sklearn.preprocessing import normalize
import torch
from sklearn.preprocessing import StandardScaler


class Mydataset(Dataset):
    def __init__(self, data_path, skip=5):
        self.data_path = data_path
        self.skip = skip
        self.sc = StandardScaler()
        df = pd.read_csv(self.data_path)
        self.colnum = df.columns.values
        label = df.iloc[:, :2]
        data = df.iloc[:, 2:]
        drop_list = label.groupby("motion").tail(skip)
        for i in drop_list.itertuples(name=None):
            label = label.drop(i[0])
        self.sc.fit(data)
        # self.data = self.sc.transform(data)
        self.data = data.values
        self.label = label.reset_index().values

    def __getitem__(self, index):
        get_index = self.label[index][0]
        input = self.data[[get_index, get_index+self.skip]]
        # input_2 = self.data[get_index+self.skip]
        # input = np.concatenate([input_1, input_2])
        output = self.data[get_index+1:get_index+self.skip]
        input = torch.from_numpy(input).float()
        output = torch.from_numpy(output).float()

        return input, output

    def get_sc(self):
        return self.sc

    def get_colnum(self):
        return self.colnum

    def __len__(self):
        return len(self.label)


class Mydataset_test(Dataset):
    def __init__(self, data_path, skip, sc: StandardScaler):
        self.data_path = data_path
        self.sc = sc
        self.skip = skip
        df = pd.read_csv(self.data_path)
        data = df.iloc[:, 2:]
        df = df.dropna(how='any')
        label = df.iloc[:, :2]

        drop_list = label.groupby("motion").tail(1)
        for i in drop_list.itertuples(name=None):
            label = label.drop(i[0])
        # self.sc.fit(data)
        # print(len(drop_list))
        # print(len(data))
        # print(len(label))
        self.data = data.values
        # self.data = self.sc.transform(data)
        self.label = label.reset_index().values
        # print("a")

    def __getitem__(self, index):
        label = self.label[index]
        get_index = self.label[index][0]
        input = self.data[[get_index, get_index+self.skip]]
        input = torch.from_numpy(input).float()
        label = torch.from_numpy(label[1:]).int()

        return input, label

    def __len__(self):
        return len(self.label)
