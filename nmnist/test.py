from __future__ import print_function

import matplotlib.pyplot as plt
import xlsxwriter
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from modelnew import *
from layersnew import *
from dataset import NMNIST


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.MSELoss(reduction='sum')(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            target_label = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target_label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=40, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=65, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default= 0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    train_dataset = NMNIST(train=True, step=steps, dt=dt,
                           path = '/nmnist/data/NMNIST_npy/Train'
                           )
    test_dataset = NMNIST(train=False, step=steps, dt=dt,
                          path = '/nmnist/data/NMNIST_npy/Test'
                          )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = densenet123()
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_path = '/nmnist/tmp/nmnist_highest.pt'
    epoch = 1

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.module.load_state_dict(checkpoint['net'])
    epoch = checkpoint['end_epoch']
    print('Model loaded.')

    modeltest = model
    test(args, modeltest, device, test_loader, epoch)


if __name__ == '__main__':
    main()
