import torch
from torch.utils.data import Dataset, DataLoader, random_split;
from torchvision import datasets
from torchvision.transforms import ToTensor
import pytorch_lightning as pl;

import glob;
import pandas as pd
import numpy as np
from torchvision.io import read_image

class Data(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.image_paths = glob.glob(img_dir + "/images/*");
        self.label_paths = glob.glob(img_dir + "/labels/*");
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label_path = self.label_paths[idx]
        image_path = self.image_paths[idx]
        image = read_image(image_path)
        # temp = int(idx)
        # while(True):
        #     label_path = self.label_paths[temp]
        #     raw_label = np.loadtxt(label_path, comments="#", delimiter=" ", unpack=False)
        #     temp += 1;
        #     if(temp >= len(self.image_paths)):
        #         temp = 0;

        #     if(len(raw_label) == 4):
        #         classes = [0, 1, 2, 3];
        #         is_ok = True
        #         for cls in classes:
        #             check = False
        #             for i in range(raw_label.shape[0]):
        #                 if raw_label[i, 0] == cls:
        #                     check = True;
        #             if not check:
        #                 is_ok = False
        #         if is_ok:
        #             break;
        
        raw_label = np.loadtxt(label_path, comments="#", delimiter=" ", unpack=False).reshape(-1, 5)
        raw_label[:, 1] = raw_label[:, 1] * image.shape[1]
        raw_label[:, 2] = raw_label[:, 2] * image.shape[2]

        # def position(p):
        #     def find_line(p1, p2):
        #         if(p1[0] - p2[0] == 0):
        #             wx = 1
        #             wy = 0
        #             b = -p1[0]
        #         else:
        #             wx = (p1[1] - p2[1]) / (p1[0] - p2[0]);
        #             wy = -1;
        #             b = p1[1] - wx * p1[0];
        #         return wx, wy, b;

        #     def check_diag(p1, p2):
        #         p3 = []
        #         p4 = []
        #         for i in range(4):
        #             if (raw_label[i, 1:3] != p1).any() and (raw_label[i, 1:3] != p2).any():
        #                 if len(p3) == 0:
        #                     p3 = raw_label[i, 1:3];
        #                 else:
        #                     p4 = raw_label[i, 1:3];

        #         wx, wy, b = find_line(p1, p2)
        #         mag1 = wx * p3[0] + wy * p3[1] + b;
        #         mag2 = wx * p4[0] + wy * p4[1] + b;
        #         return (mag1 * mag2 < 0)
            
        #     res = 0;
        #     for i in range(4):
        #         other = raw_label[i, 1:3]
        #         if (other != p).any():
        #             if check_diag(p, other) == True:
        #                 vec = other - p
        #                 if(vec[1] > 0):
        #                     res += 2;
        #                 if(vec[0] > 0):
        #                     res += 1;

        #     return res;

        # for i in range(4):
        #     label[position(raw_label[i, 1:3])] = raw_label[i, 1:3];
        
        # print(idx, label)
        cls = np.array(raw_label[:, 0] + 1, dtype=np.int64)
        #cls[:] = 1;
        box = np.repeat(raw_label[:, 1:3], 2, axis=1)
        box[:, 2], box[:, 1] = np.copy(box[:, 1]), np.copy(box[:, 2])
        box[:, 0:2] = np.copy(box[:, 0:2] - 30);
        box[:, 2:] = np.copy(box[:, 2:] + 30)
        label = {
            'boxes': torch.from_numpy(box),
            'labels': torch.from_numpy(cls)
        }
        return image, label

class DataModule(pl.LightningDataModule):
    def collate_fn(self, data):
        image = []
        label = []
        for x, y in data:
            image.append(x);
            label.append(y);
        return image, label
    def __init__(self, batch_size, num_workers):
        super().__init__();
        self.batch_size = batch_size;
        self.num_workers = num_workers;
    def setup(self, stage):
        self.train_data, self.validate_data, self.test_data = Data("../card-id-Detect4Corner-2/train"), Data("../card-id-Detect4Corner-2/valid"), Data("../card-id-Detect4Corner-2/test");
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = True, collate_fn=self.collate_fn);
    def val_dataloader(self):
        return DataLoader(self.validate_data, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False, collate_fn=self.collate_fn);
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False, collate_fn=self.collate_fn);
