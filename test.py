from data import MydataModule
from options import Options
from model import Frame_predict
from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    opt = Options().parse()

    data = MydataModule(opt)

    # test_data = data.test_dataloader()

    model = Frame_predict(opt, data)
    model.load_from_checkpoint(
        checkpoint_path=opt.weight)

    # model = Frame_predict.load_from_checkpoint(
    #     checkpoint_path=opt.weight, hparams_file=opt.yaml)

    trainer = Trainer(gpus=1)

    data.setup("test")
    test_data = data.test_dataloader()
    # for i, j in enumerate(test_data):
    #     print(i)

    trainer.test(model, test_data)


if __name__ == "__main__":
    main()
