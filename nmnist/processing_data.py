from __future__ import print_function
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from modelnew import *
from layersnew import *
from torch.utils.data import Dataset
import pandas as pd
import cv2, glob

def main():
    src_path_train = '/N-MNIST/Train'
    src_path_test = '/N-MNIST/Test'
    save_path_train = '/nmnist/data/NMNIST_npy/Train'
    save_path_test = '/nmnist/data/NMNIST_npy/Test'
    win = steps * dt
    len_train = 60000
    len_test = 10000
    eventflow_train = np.zeros(shape=(len_train, 2, 34, 34, steps))
    label_train = np.zeros(shape=(len_train, 10))
    eventflow_test = np.zeros(shape=(len_test, 2, 34, 34, steps))
    label_test = np.zeros(shape=(len_test, 10))
    filenum_train = 0
    filenum_test = 0
    for num in range(10):
        dir = os.path.join(src_path_train, str(num))
        files = os.listdir(dir)
        for file in files:
            file_dir = os.path.join(dir, file)
            f = open(file_dir, 'rb')
            raw_data = np.fromfile(f, dtype=np.uint8)
            f.close()
            raw_data = np.uint32(raw_data)

            all_y = raw_data[1::5]
            all_x = raw_data[0::5]
            all_p = (raw_data[2::5] & 128) >> 7
            all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
            all_ts = np.uint32(np.around(all_ts / 1000))

            win_indices = np.where(all_ts < win)
            win_indices = win_indices[0]
            for i in range(len(win_indices)):
                index = int(win_indices[i])
                eventflow_train[filenum_train, all_p[index], all_x[index], all_y[index], int(all_ts[index] / dt)] = 1
            label_train[filenum_train] = np.eye(10)[num]

            filenum_train += 1

        print("Train, Done file:" + str(num))

    if save_path_train is not None:
        np.save('/nmnist/data/NMNIST_npy/Train/data.npy', eventflow_train)
        np.save('/nmnist/data/NMNIST_npy/Train/label.npy', label_train)


    for num in range(10):
        dir = os.path.join(src_path_test, str(num))
        files = os.listdir(dir)
        for file in files:
            file_dir = os.path.join(dir, file)
            f = open(file_dir, 'rb')
            raw_data = np.fromfile(f, dtype=np.uint8)
            f.close()
            raw_data = np.uint32(raw_data)

            all_y = raw_data[1::5]
            all_x = raw_data[0::5]
            all_p = (raw_data[2::5] & 128) >> 7
            all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
            all_ts = np.uint32(np.around(all_ts / 1000))

            win_indices = np.where(all_ts < win)
            win_indices = win_indices[0]
            for i in range(len(win_indices)):
                index = int(win_indices[i])
                eventflow_test[filenum_test, all_p[index], all_x[index], all_y[index], int(all_ts[index] / dt)] = 1
            label_test[filenum_test] = np.eye(10)[num]

            filenum_test += 1

        print("Test, Done file:" + str(num))

    if save_path_test is not None:
        np.save('/nmnist/data/NMNIST_npy/Test/data.npy', eventflow_test)
        np.save('/nmnist/data/NMNIST_npy/Test/label.npy', label_test)

if __name__ == '__main__':
    main()