import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from utils.dataset import BasicDataSet
from model.net import MyNet
from torch.utils.data import DataLoader, random_split

dir_img = './data/train_img/'
dir_label = './data/train_label/'


def main():
    if torch.cuda.is_available():
        USE_CUDA = True
        print('Using GPU, %i devices.' % torch.cuda.device_count())
    else:
        USE_CUDA = False

    num_classes = 2
    dateSet = BasicDataSet(dir_img, dir_label, 1)

    dataloader = DataLoader(dateSet, batch_size=2, shuffle=True, num_workers=4)

    model = MyNet(n_channel=1)
    if USE_CUDA:
        model.cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    lr_schedular = optim.lr_scheduler.StepLR(model_optim, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        iterator = tqdm(dataloader, unit='img')
        for i, data in enumerate(iterator):
            iterator.set_description('Epoch %i/%i' % (epoch + 1, num_epochs))

            img = data['image']
            label = data['label']

            if USE_CUDA:
                img = img.cuda()
                label = label.cuda()

            model_optim.zero_grad()

            outputs = model(img)
            loss = criterion(outputs, label)

            epoch_loss += loss.item()

            loss.backward()
            model_optim.step()
            lr_schedular.step()

            iterator.set_postfix(loss='{}'.format(loss.data))


if __name__ == '__main__':
    main()

