import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, opt):
        super(Unet, self).__init__()
        self.skip = opt.skip - 1

        self.nnmodel = nn.Sequential()
        self.nnmodel.add_module("flat1", nn.Flatten())  # (B,20*63)
        self.nnmodel.add_module("fc1", nn.Linear(2*63, 20*63))
        self.nnmodel.add_module('relu3', nn.ReLU(inplace=True))
        self.nnmodel.add_module('drop1', nn.Dropout(p=0.5))
        self.nnmodel.add_module("fc2", nn.Linear(20*63, 20*63*self.skip))
        self.nnmodel.add_module('relu4', nn.ReLU(inplace=True))
        self.nnmodel.add_module('drop2', nn.Dropout(p=0.5))
        self.nnmodel.add_module("fc3", nn.Linear(
            20*63*self.skip, 63*self.skip))
        # self.model.add_module("fc2", nn.Linear(630, 63))

    def forward(self, x):
        b = x.size()[0]
        x = self.nnmodel(x)
        return x.view(self.skip, b, 63)
