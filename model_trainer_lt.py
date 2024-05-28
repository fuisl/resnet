import os
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import torchmetrics as tm

from resnet import ResNetGlobalMaxPool, resnet34_global_max_pool


import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
import torchvision


class ImageClassificationTrainer(LightningModule):
    """
    """

    def __init__(
        self,
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = resnet34_global_max_pool()
        self.accuracy = tm.Accuracy(task='multiclass', num_classes=3)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        out = self.forward(images)
        loss = nn.CrossEntropyLoss()(out, targets)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss
        

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        out = self.forward(images)
        loss = nn.CrossEntropyLoss()(out, targets)
        accuracy = self.accuracy(out, targets)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_acc', accuracy, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        out = self.forward(images)
        loss = nn.CrossEntropyLoss(task='multiclass', num_classes=3)(out, targets)
        accuracy = self.accuracy(out, targets)
        self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('test_acc', accuracy, prog_bar=True, on_step=True, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 20))
        return [optimizer], [scheduler]

def main() -> None:
    pl.seed_everything()

    model = ImageClassificationTrainer()
 
    # if os.path.isfile(args.resume):
    #     print('=> loading checkpoint: {}'.format(args.resume))
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     model.model.load_state_dict(checkpoint['state_dict'])
    #     print('=> loaded checkpoint: {}'.format(args.resume))
    pass



if __name__ == "__main__":
    main()