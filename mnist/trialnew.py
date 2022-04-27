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


def kl_div_p12(p1, p2):
    kl_12 = F.kl_div(F.log_softmax(p1, dim=-1), F.softmax(p2, dim=-1), reduction='none').mean()
    kl_21 = F.kl_div(F.log_softmax(p2, dim=-1), F.softmax(p1, dim=-1), reduction='none').mean()
    loss = (kl_12 + kl_21) / 2.
    return loss


def train(args, model, device, train_loader, optimizer, epoch, train_acc, train_ls):
    model.train()
    train_loss = 0
    minibatch_num = 0
    correct = 0
    for param_group in optimizer.param_groups:
        print('current learning rate', param_group['lr'])
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
        data = data.permute(1, 2, 3, 4, 0)
        output1, output2 = model(data)
        loss1 = F.cross_entropy(output1, target)
        loss2 = loss1 + F.cross_entropy(output2, target)
        loss = 0.5 * loss2 + 0.5 * kl_div_p12(output1, output2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()
        minibatch_num = minibatch_num + 1
        pred1 = output1.argmax(dim=1, keepdim=True)
        pred2 = output2.argmax(dim=1, keepdim=True)
        correct += pred1.eq(target.view_as(pred1)).sum().item()
        correct += pred2.eq(target.view_as(pred2)).sum().item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data / steps), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    correct = correct / 2
    train_acc.append(100. * correct / len(train_loader.dataset))
    train_ls.append(train_loss / minibatch_num)

    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
          .format(train_loss / minibatch_num, correct, len(train_loader.dataset),
                  100. * correct / len(train_loader.dataset)))


def test(args, model, device, test_loader, epoch, test_acc, test_ls):
    model.eval()
    test_loss = 0
    correct = 0
    totalspike = 0
    batch_idx = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
            data = data.permute(1, 2, 3, 4, 0)
            output = model(data, batch_idx, epoch)
            batch_idx += 1
            totalspike += (output>Vth).float().sum().item()
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('average spike number', totalspike / len(test_loader.dataset))
    acc = 100. * correct / len(test_loader.dataset)
    test_acc.append(acc)
    test_ls.append(test_loss)

    if epoch > 5:
        if acc == torch.max(torch.Tensor(test_acc)):
            if (args.save_model):
                state = {
                    'net': model.module.state_dict(),
                    'end_epoch': epoch,
                }
                torch.save(state, "/mnist/tmp/mnist_highest.pt")

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))


def adjust_learning_rate(args, optimizer, epoch):
    if epoch > 60:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001
    elif epoch > 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00005
    elif epoch > 40:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    elif epoch > 30:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005
    elif epoch > 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
    elif epoch > 10:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005

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

    checkpoint_path = '/mnist/tmp/mnist.pt'
    start_epoch = 1
    test_acc = []
    train_acc = []
    test_ls = []
    train_ls = []

    if os.path.isdir(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['end_epoch'] + 1
        print('Model loaded.')
        print('########### Starting epoch index after loading: %d ###########'
              % start_epoch)

    for epoch in range(start_epoch, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch, train_acc, train_ls)
        test(args, model, device, test_loader, epoch, test_acc, test_ls)

        if epoch % 5 == 0:
            if (args.save_model):
                state = {
                    'net': model.module.state_dict(),
                    'end_epoch': epoch,
                }
                torch.save(state, "/mnist/tmp/mnist.pt")

                workbook = xlsxwriter.Workbook('testaccuracy.xlsx')
                worksheet = workbook.add_worksheet()
                for col, data in enumerate(test_acc):
                    worksheet.write_number(col, 0, data)
                for col, data in enumerate(train_acc):
                    worksheet.write_number(col, 1, data)
                for col, data in enumerate(test_ls):
                    worksheet.write_number(col, 2, data)
                for col, data in enumerate(train_ls):
                    worksheet.write_number(col, 3, data)
                workbook.close()


if __name__ == '__main__':
    main()