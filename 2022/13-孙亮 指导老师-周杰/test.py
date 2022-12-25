import argparse
import shutil

import torchvision.models

import train_operation as operation
import model_genotype as genotype
import models

import torch
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import sys
import numpy as np
import logging
import time
import os
# =================================================================================
# set arguments for Terminal input
parser = argparse.ArgumentParser("Train an architecture created by genome")
parser.add_argument('--demo', type=int, default=1, help='select number of demo architecture')
parser.add_argument('--weight_file', type=str, default='weight1.pth', help='file path of weight')

parser.add_argument('--save', type=str, default='train', help='experiment name')

parser.add_argument('--n_blocks', type=int, default=5, help='number of blocks in a cell')
parser.add_argument('--n_nodes', type=int, default=4, help='number of nodes per phases')

parser.add_argument('--init_channels', type=int, default=36, help='channels of filters for first cell')
parser.add_argument('--layers', type=int, default=20, help='number of layers of the networks')

parser.add_argument('--device', type=str, default='cuda:0', help='GPU/CPU device selected')

args = parser.parse_args()


def cifar10_transforms():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    return valid_transform


def infer(valid_loader, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    logging.info('valid acc %f', 100. * correct / total)

    return test_loss / total, acc

demo1 = [6, 1, 3, 0,
         1, 1, 3, 1,
         0, 3, 6, 2,
         6, 0, 6, 1,
         0, 1, 7, 4,
         2, 0, 2, 0,
         0, 0, 2, 1,
         2, 1, 7, 0,
         3, 3, 9, 3,
         2, 1, 7, 3]

demo2 = [8, 0, 8, 0,
         1, 1, 3, 0,
         3, 1, 2, 3,
         8, 1, 6, 3,
         6, 3, 1, 3,
         2, 0, 2, 0,
         0, 0, 2, 1,
         2, 1, 7, 0,
         3, 3, 3, 0,
         3, 1, 4, 1]

device = args.device

if __name__ == '__main__':
    seed = args.seed
    torch.cuda.set_device(device)
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    assert args.demo <= 2
    genome = []
    if args.demo == 1:
        genome = demo1
    elif args.demo == 2:
        genome = demo2


    valid_transform = cifar10_transforms()
    valid_set = torchvision.datasets.CIFAR10(root='../data/', train=False, transform=valid_transform, download=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=96, shuffle=False, num_workers=2, pin_memory=True)

    geno = genotype.decode(genome)
    net = models.NetworkCIFAR(C=args.init_channels, num_classes=10, layers=args.layers, auxiliary=True,
                              genotype=geno)
    net.load(torch.load(args.weight_file))
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    n_flops, n_params = operation.get_flops(net, device)
    valid_loss, valid_acc = infer(valid_loader, net, criterion)

    print(f"FLOPs: {n_flops / 1e6}M")
    print(f"Params: {n_params / 1e6}M")
    print(f"Accuracy: {valid_acc}")

