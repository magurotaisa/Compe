from data import MydataModule
from options import Options
from model import Frame_predict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    opt = Options().parse()

    data = MydataModule(opt)

    model = Frame_predict(opt, data)

    # data.setup()
    # train = data.train_dataloader()
    # for i in train:
    #     input = i[0]
    #     y = model(input)

    #     print(len(y))
    #     print(len(i[1]))

    #     break

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="sample-max-{epoch:02d}-{val_loss:.3f}",
        save_top_k=3,
        mode="min",
    )

    trainer = Trainer(gpus=1, max_epochs=opt.epoch,
                      callbacks=[checkpoint_callback])

    trainer.fit(model, data)

    trainer.test(ckpt_path="best", dataloaders=data)


if __name__ == "__main__":
    main()
