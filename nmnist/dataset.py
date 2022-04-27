import numpy as np
import torch
import os
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms, utils
import cv2, glob


class NMNIST(Dataset):
    def __init__(self, train, step, dt, path=None):
        super(NMNIST, self).__init__()
        self.step = step
        self.path = path
        self.train = train
        self.dt = dt
        self.win = step * dt
        self.len = 60000
        if train == False:
            self.len = 10000
        self.eventflow = np.zeros(shape=(self.len, 2, 34, 34, self.step))
        self.label = np.zeros(shape=(self.len, 10))
        
        if path is not None:
            self.eventflow = np.load(path + '/data.npy')
            self.label = np.load(path+'/label.npy')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.eventflow[idx, ...].astype(np.float32)     # 某些情况下可能对数据格式有要求（MSELoss）
        y = self.label[idx].astype(np.float32)                
        return (x, y)

    def preprocessing(self, src_path, save_path=None):
        filenum = 0
        for num in range(10):
            dir = os.path.join(src_path, str(num))
            files = os.listdir(dir)
            for file in files:
                file_dir = os.path.join(dir, file)
                f = open(file_dir, 'rb')
                raw_data = np.fromfile(f, dtype=np.uint8)
                f.close()
                raw_data = np.uint32(raw_data)

                all_y = raw_data[1::5]
                all_x = raw_data[0::5]
                all_p = (raw_data[2::5] & 128) >> 7 #bit 7
                all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
                all_ts = np.uint32(np.around(all_ts / 1000))

                win_indices = np.where(all_ts < self.win)
                win_indices = win_indices[0]
                for i in range(len(win_indices)): 
                    index = int(win_indices[i])
                    polar = 0
                    self.eventflow[filenum, polar, all_x[index], all_y[index], int(all_ts[index] / self.dt)] = 1
                self.label[filenum] = np.eye(10)[num]

                filenum += 1
                
            print("Done file:" + str(num))
        
        if save_path is not None:
            field = "Train" if self.train else "Test"
            np.save("./data/NMNIST_npy/"+field+"/data.npy", self.eventflow)
            np.save("./data/NMNIST_npy/"+field+"/label.npy", self.label)


