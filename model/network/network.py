import torch.nn as nn
import torch


class OutNetwork(nn.Module):
    def __init__(self):
        super(OutNetwork, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module("fc1", nn.Linear(400, 400))
        self.model.add_module('relu1', nn.ReLU(inplace=True))
        self.model.add_module('drop1', nn.Dropout(p=0.5))
        self.model.add_module("fc2", nn.Linear(400, 63))

    def forward(self, x):
        x = self.model(x)
        return x


class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module("conv1", nn.Conv1d(
            2, 5, 3, stride=3))  # (16,5,21)
        self.model.add_module('relu1', nn.ReLU(inplace=True))
        self.model.add_module("conv2", nn.Conv1d(
            5, 10, 1, stride=1))  # (16,10,21)
        self.model.add_module('relu2', nn.ReLU(inplace=True))
        self.model.add_module("conv3", nn.Conv1d(
            10, 20, 1, sa4tride=1))  # (16,10,21)
        self.model.add_module('relu3', nn.ReLU(inplace=True))
        self.model.add_module("flat1", nn.Flatten())  # (16,105)
        self.model.add_module("fc1", nn.Linear(20*21, 630))
        self.model.add_module('relu4', nn.ReLU(inplace=True))
        self.model.add_module('drop1', nn.Dropout(p=0.5))
        self.model.add_module("fc2", nn.Linear(630, 63))

    def forward(self, x):
        x = self.model(x)
        # print(x.size())
        return x


class Convnet_v1(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module("conv1", nn.Conv1d(
            2, 5, 3, stride=3))  # (16,5,21)
        self.model.add_module('relu1', nn.ReLU(inplace=True))
        self.model.add_module("conv2", nn.Conv1d(
            5, 10, 1, stride=1))  # (16,10,21)
        self.model.add_module('relu2', nn.ReLU(inplace=True))
        self.model.add_module("conv3", nn.Conv1d(
            10, 20, 1, stride=1))  # (16,10,21)
        self.model.add_module('relu3', nn.ReLU(inplace=True))
        self.model.add_module("flat1", nn.Flatten())  # (16,105)
        self.model.add_module("fc1", nn.Linear(20*21, 630))
        self.model.add_module('relu4', nn.ReLU(inplace=True))
        self.model.add_module('drop1', nn.Dropout(p=0.5))
        self.model.add_module("fc2", nn.Linear(630, 63))

    def forward(self, x):
        x = self.model(x)
        # print(x.size())
        return x


class PreNetwork(nn.Module):
    def __init__(self):
        super(PreNetwork, self).__init__()
        self.premodel = nn.Sequential()
        self.premodel.add_module("fc1", nn.Linear(63, 126))
        # self.premodel.add_module('relu1', nn.ReLU(inplace=True))
        self.premodel.add_module('drop1', nn.Dropout(p=0.5))
        self.premodel.add_module("fc2", nn.Linear(126, 126))
        # self.premodel.add_module('relu2', nn.ReLU(inplace=True))
        self.premodel.add_module('drop2', nn.Dropout(p=0.5))
        self.premodel.add_module("fc3", nn.Linear(126, 100))
        self.premodel.add_module('relu3', nn.ReLU(inplace=True))
        self.premodel.add_module('drop3', nn.Dropout(p=0.5))

    def forward(self, x):
        x = self.premodel(x)
        return x


class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        # self.premodel_1 = PreNetwork()
        # self.premodel_2 = PreNetwork()
        self.out_num = opt.skip - 1
        # self.outputmodel = list()
        # self.convnet = Convnet()
        # for _ in range(self.out_num):
        #     self.outputmodel.append(OutNetwork().to('cuda'))
        # for _ in range(self.out_num):
        #     self.outputmodel.append(Convnet().to('cuda'))
        self.model1 = Convnet()
        self.model2 = Convnet()
        self.model3 = Convnet()
        self.model4 = Convnet()

    def forward(self, x):
        # x:(16,2,63)
        # x_0 = self.convnet(x)
        # x = x.transpose(0, 1)
        # # x:(2,16,63)
        # x_1 = self.premodel_1(x[0])
        # x_2 = self.premodel_2(x[1])
        # out_list = list()
        x_1 = self.model1(x)
        x_2 = self.model2(x)
        x_3 = self.model3(x)
        x_4 = self.model4(x)
        # for out in self.outputmodel:
        #     out_list.append(out(x))
        # print(out_list)

        return torch.stack([x_1, x_2, x_3, x_4])


class Network_45(nn.Module):
    def __init__(self, opt):
        super(Network_45, self).__init__()
        # self.premodel_1 = PreNetwork()
        # self.premodel_2 = PreNetwork()
        self.out_num = opt.skip - 1
        # self.outputmodel = list()
        # self.convnet = Convnet()
        # for _ in range(self.out_num):
        #     self.outputmodel.append(OutNetwork().to('cuda'))
        # for _ in range(self.out_num):
        #     self.outputmodel.append(Convnet().to('cuda'))
        self.model1 = Convnet()
        self.model2 = Convnet()
        self.model3 = Convnet()
        self.model4 = Convnet()
        self.model5 = Convnet()
        self.model6 = Convnet()
        self.model7 = Convnet()
        self.model8 = Convnet()
        self.model9 = Convnet()
        self.model10 = Convnet()
        self.model11 = Convnet()
        self.model12 = Convnet()
        self.model13 = Convnet()
        self.model14 = Convnet()
        self.model15 = Convnet()
        self.model16 = Convnet()
        self.model17 = Convnet()
        self.model18 = Convnet()
        self.model19 = Convnet()
        self.model20 = Convnet()
        self.model21 = Convnet()
        self.model22 = Convnet()
        self.model23 = Convnet()
        self.model24 = Convnet()
        self.model25 = Convnet()
        self.model26 = Convnet()
        self.model27 = Convnet()
        self.model28 = Convnet()
        self.model29 = Convnet()
        self.model30 = Convnet()
        self.model31 = Convnet()
        self.model32 = Convnet()
        self.model33 = Convnet()
        self.model34 = Convnet()
        self.model35 = Convnet()
        self.model36 = Convnet()
        self.model37 = Convnet()
        self.model38 = Convnet()
        self.model39 = Convnet()
        self.model40 = Convnet()
        self.model41 = Convnet()
        self.model42 = Convnet()
        self.model43 = Convnet()
        self.model44 = Convnet()

    def forward(self, x):
        x_1 = self.model1(x)
        x_2 = self.model2(x)
        x_3 = self.model3(x)
        x_4 = self.model4(x)
        x_5 = self.model5(x)
        x_6 = self.model6(x)
        x_7 = self.model7(x)
        x_8 = self.model8(x)
        x_9 = self.model9(x)
        x_10 = self.model10(x)
        x_11 = self.model11(x)
        x_12 = self.model12(x)
        x_13 = self.model13(x)
        x_14 = self.model14(x)
        x_1 = self.model1(x)
        x_2 = self.model2(x)
        x_3 = self.model3(x)
        x_4 = self.model4(x)
        x_5 = self.model5(x)
        x_6 = self.model6(x)
        x_7 = self.model7(x)
        x_8 = self.model8(x)
        x_9 = self.model9(x)
        x_10 = self.model10(x)
        x_11 = self.model11(x)
        x_12 = self.model12(x)
        x_13 = self.model13(x)
        x_14 = self.model14(x)
        x_15 = self.model15(x)
        x_16 = self.model16(x)
        x_17 = self.model17(x)
        x_18 = self.model18(x)
        x_19 = self.model19(x)
        x_20 = self.model20(x)
        x_21 = self.model21(x)
        x_22 = self.model22(x)
        x_23 = self.model23(x)
        x_24 = self.model24(x)
        x_25 = self.model25(x)
        x_26 = self.model26(x)
        x_27 = self.model27(x)
        x_28 = self.model28(x)
        x_29 = self.model29(x)
        x_30 = self.model30(x)
        x_31 = self.model31(x)
        x_32 = self.model32(x)
        x_33 = self.model33(x)
        x_34 = self.model34(x)
        x_35 = self.model35(x)
        x_36 = self.model36(x)
        x_37 = self.model37(x)
        x_38 = self.model38(x)
        x_39 = self.model39(x)
        x_40 = self.model40(x)
        x_41 = self.model41(x)
        x_42 = self.model42(x)
        x_43 = self.model43(x)
        x_44 = self.model44(x)
        # x:(16,2,63)
        # x_0 = self.convnet(x)
        # x = x.transpose(0, 1)
        # # x:(2,16,63)
        # x_1 = self.premodel_1(x[0])
        # x_2 = self.premodel_2(x[1])
        # out_list = list()

        # for out in self.outputmodel:
        #     out_list.append(out(x))
        # print(out_list)

        return torch.stack([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31, x_32, x_33, x_34, x_35, x_36, x_37, x_38, x_39, x_40, x_41, x_42, x_43, x_44])


class Network_15(nn.Module):
    def __init__(self, opt):
        super(Network_15, self).__init__()
        # self.premodel_1 = PreNetwork()
        # self.premodel_2 = PreNetwork()
        self.out_num = opt.skip - 1
        # self.outputmodel = list()
        # self.convnet = Convnet()
        # for _ in range(self.out_num):
        #     self.outputmodel.append(OutNetwork().to('cuda'))
        # for _ in range(self.out_num):
        #     self.outputmodel.append(Convnet().to('cuda'))
        self.model1 = Convnet()
        self.model2 = Convnet()
        self.model3 = Convnet()
        self.model4 = Convnet()
        self.model5 = Convnet()
        self.model6 = Convnet()
        self.model7 = Convnet()
        self.model8 = Convnet()
        self.model9 = Convnet()
        self.model10 = Convnet()
        self.model11 = Convnet()
        self.model12 = Convnet()
        self.model13 = Convnet()
        self.model14 = Convnet()

    def forward(self, x):
        # x:(16,2,63)
        # x_0 = self.convnet(x)
        # x = x.transpose(0, 1)
        # # x:(2,16,63)
        # x_1 = self.premodel_1(x[0])
        # x_2 = self.premodel_2(x[1])
        # out_list = list()
        x_1 = self.model1(x)
        x_2 = self.model2(x)
        x_3 = self.model3(x)
        x_4 = self.model4(x)
        x_5 = self.model5(x)
        x_6 = self.model6(x)
        x_7 = self.model7(x)
        x_8 = self.model8(x)
        x_9 = self.model9(x)
        x_10 = self.model10(x)
        x_11 = self.model11(x)
        x_12 = self.model12(x)
        x_13 = self.model13(x)
        x_14 = self.model14(x)
        # for out in self.outputmodel:
        #     out_list.append(out(x))
        # print(out_list)

        return torch.stack([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14])


def create_model(opt):
    if opt.skip == 5:
        model = Network(opt)
    elif opt.skip == 15:
        model = Network_15(opt)
    elif opt.skip == 45:
        model = Network_45(opt)
    return model
