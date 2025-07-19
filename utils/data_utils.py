import os
import gzip
import torch
import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)

class MyDatasetPro(Dataset):
    def __init__(self, img_dir, data_array):
        self.img_dir = img_dir
        self.data_array = data_array

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, index):
        sid = self.data_array[index, 0]
        file_name = f"{int(sid)}.npy.gz"
        img_path = os.path.join(self.img_dir, file_name)

        position = torch.tensor(self.data_array[index, 1:3], dtype=torch.float32)
        veg = torch.tensor(self.data_array[index, 3], dtype=torch.float32).unsqueeze(0)

        with gzip.open(img_path, "r") as f:
            img = torch.tensor(np.load(f), dtype=torch.float32)

    
        return img, position, veg

def get_loader(args):
    img_dir = r"./data/RS_sample"
    data = pd.read_csv(r"./data/dataset.csv")

    # Split dataset into training (type=0) and eval (type=1)
    train_data = data[data['type'] == 0]
    test_data = data[data['type'] == 1]

    # Select relevant columns: sid, longitude, latitude, gvi
    cols = ['sid', 'longitude', 'latitude', 'gvi']
    train_array = train_data[cols].values
    test_array = test_data[cols].values

    # Normalize coordinates to [0, 1] range
    lon_min, lon_max = 102, 130.5
    lat_min, lat_max = 22, 46.5

    train_array[:, 1] = (train_array[:, 1] - lon_min) / (lon_max - lon_min)
    test_array[:, 1] = (test_array[:, 1] - lon_min) / (lon_max - lon_min)

    train_array[:, 2] = (train_array[:, 2] - lat_min) / (lat_max - lat_min)
    test_array[:, 2] = (test_array[:, 2] - lat_min) / (lat_max - lat_min)

    # Create dataset instances
    train_dataset = MyDatasetPro(img_dir, train_array)
    test_dataset = MyDatasetPro(img_dir, test_array)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size,
        num_workers=5,
        pin_memory=True
    )

    eval_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size,
        num_workers=5,
        pin_memory=True
    )

    return train_loader, eval_loader
