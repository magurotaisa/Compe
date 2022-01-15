from pytorch_lightning import LightningModule
from data import MydataModule
import torch.nn as nn
import torch
import numpy as np
from model.network import create_model, Unet
import pandas as pd


class Frame_predict(LightningModule):
    def __init__(self, opt, data: MydataModule) -> None:
        super().__init__()

        self.opt = opt
        self.save_hyperparameters()
        # self.net = create_model(opt)
        self.net = Unet(opt)
        self.loss = nn.MSELoss()
        # self.sc = data.get_sc()
        self.test_list = list()
        self.col_num = data.get_colnum()
        # self.batch = opt.batch

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        input = batch[0]
        # print(input.shape)
        label = batch[1].transpose(0, 1)

        output = self(input)
        loss_list = list()

        for i, y in enumerate(zip(output, label)):
            loss = self.loss(y[0], y[1])
            loss_list.append(loss)
            self.log(f"train/loss_opse{i}", loss,
                     on_step=False, on_epoch=True)

        total_loss = torch.stack(loss_list).mean()

        self.log("train/total_loss", total_loss,
                 on_step=False, on_epoch=True)

        return {"loss": total_loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("epoch_loss", avg_loss,
                 on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        input = batch[0]
        label = batch[1].transpose(0, 1)

        output = self(input)
        loss_list = list()

        for _, y in enumerate(zip(output, label)):
            loss = self.loss(y[0], y[1])
            loss_list.append(loss)

        total_loss = torch.stack(loss_list).mean()

        return {"loss": total_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("val_loss", avg_loss,
                 on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        input = batch[0]
        label = batch[1].to('cpu').detach().numpy().copy()
        output = self(input).to('cpu').detach().numpy().copy()
        output = output.transpose(1, 0, 2)
        input = input.to('cpu').detach().numpy().copy()
        for input, outputs, labels in zip(input, output, label):
            if labels[1] == 1:
                col = np.append(labels, input[0])
                self.test_list.append(col.tolist())
            for out in outputs:
                labels[1] = labels[1] + 1
                col = np.append(labels, out)
                self.test_list.append(col)
            labels[1] = labels[1] + 1
            col = np.append(labels, input[1])
            self.test_list.append(col)

    def test_epoch_end(self, _):
        test_list = np.array(self.test_list)
        # test_list[:, 2:] = self.sc.inverse_transform(test_list[:, 2:])
        df = pd.DataFrame(data=test_list,
                          columns=self.col_num,
                          dtype='float')
        df = df.astype({'motion': int, 'frame_id': int})
        df.to_csv(self.opt.save, index=False)

        self.test_list = list()

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.net.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return optim
