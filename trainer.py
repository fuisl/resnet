from data import OCT_Data
from model_trainer_lt import ImageClassificationTrainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
    model = ImageClassificationTrainer(
        lr = 1e-4,
        )

    dm = OCT_Data(batch_size=256, workers=15)


    trainer = Trainer(
        callbacks = [
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(filename='{epoch}-{val_acc:.4f}', save_top_k=5, monitor='val_acc', mode='max'),
        ], 
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        default_root_dir='checkpoint/classification',

        deterministic=False, 
        max_epochs=50, 

        devices = [0],
        # precision=16,
        # strategy='ddp',        
    )

    if False:
        trainer.test(model, dm)
    else:
        trainer.fit(model, dm)