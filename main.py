import timm
from digitrec.trainer import Trainer


if __name__=="__main__":
    model = timm.create_model("resnet10t", num_classes=10, in_chans=1)
    trainer = Trainer(model=model, optimizer="adam")

    trainer.fit(epochs=10)