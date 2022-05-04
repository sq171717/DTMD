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
from tensorboardX import SummaryWriter


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    batch_idx = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
            data = data.permute(1, 2, 3, 4, 0)
            output = model(data, batch_idx, epoch)
            batch_idx += 1
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=80, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=70, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=300, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--drop-rate', type=float, default=0.2, help='dropout rate')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    args.device = device

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    test_set = datasets.MNIST('../data', train=False,
                       transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

    model = Net(args)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters', pytorch_total_params)
    print('total trainable parameters', pytorch_total_train_params)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0001)

    checkpoint_path = '/mnist/tmp/mnist_highest.pt'
    epoch = 1

    checkpoint = torch.load(checkpoint_path)
    model.module.load_state_dict(checkpoint['net'])
    epoch = checkpoint['end_epoch']
    print('Model loaded.')

    test(args, model, device, test_loader, epoch)


if __name__ == '__main__':
    main()
