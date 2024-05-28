from torch.utils.data import Dataset
from PIL import Image
import glob
from torchvision import transforms
import numpy as np
import cv2

from typing import Optional
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
import pytorch_lightning as pl
import cv2
import numpy as np
from tqdm import tqdm

import os
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = glob.glob(self.root_dir + "/**/*.jpg", recursive=True)
        self.img_paths.sort()
        print('Found {} images for training'.format(len(self.img_paths)))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        fetch_img = self.img_paths[index]

        img = Image.open(fetch_img)

        transformer = transforms.Compose([
            transforms.RandomCrop((256, 256)),   # Resize the image to 256x256
            transforms.ToTensor()            # Convert the image to a tensor
        ])

        img = transformer(img)

        return img, int(os.path.basename(os.path.dirname(os.path.dirname(fetch_img))))
    
    
# dataset = ImageDataset('/data1tb/haiduong/n2n/dataset/train')
# from torch.utils.data import DataLoader
# train_loader = DataLoader(dataset=dataset,
#                             num_workers=8,
#                             batch_size=10,
#                             shuffle=True,
#                             pin_memory=False,
#                             drop_last=True)

# # for img, classes in train_loader:
# #     continue
# # dataset.__getitem__(0)

class OCT_Data(pl.LightningDataModule):
    def __init__(self,         
        batch_size: int = 10,
        workers: int = 5,
        train_data: str = "/home/fuisloy/data1tb/resnet/dataset/train",
        val_data: str = "/home/fuisloy/data1tb/resnet/dataset/train",
        test_data: str = "/home/fuisloy/data1tb/resnet/dataset/train",
        ):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = ImageDataset(root_dir=self.train_data)
            self.val_dataset = ImageDataset(root_dir=self.val_data)
        if stage == "test" or stage is None:
            self.test_dataset = ImageDataset(root_dir=self.test_data)

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers,
            persistent_workers=True, 
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader( 
            dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
            persistent_workers=True, 
        )
        return val_loader
    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    # transforms = Compose([
    #                 RandomCrop(512, 512, p=1.0),
    #                 ], bbox_params=BboxParams(format='pascal_voc', min_visibility=0.85, label_fields=[]))

    # zalo_data = ImageDetectionDataset(dataframe=pd.read_csv('za_traffic_2020/traffic_train/annotation.csv'),
    #                                   image_dir='za_traffic_2020/traffic_train/images',
    #                                   transforms=transforms,
    #                                   )
    # # zalo_data.__getitem__(0)
    # for image, label in tqdm(zalo_data):
    #     pass
    # pass

    dm = OCT_Data()
    dm.setup()
    trainloader = dm.train_dataloader()
    for img, label in trainloader:
        print()
        pass 