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
from model.net import DenseUnet
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import accuracy_score


dir_img = './data/train_img/'
dir_label = './data/train_label/'
test_img='./data/test_img/'
test_label='./data/test_label/'

def main():
    if torch.cuda.is_available():
        USE_CUDA = True
        print('Using GPU, %i devices.' % torch.cuda.device_count())
    else:
        USE_CUDA = False

    num_classes = 2
    dateSet = BasicDataSet(dir_img, dir_label, 1)
    testSet = BasicDataSet(test_img,test_label,1)
    dataloader = DataLoader(dateSet, batch_size=2, shuffle=True, num_workers=4)

    model = DenseUnet()
    if USE_CUDA:
        model=model.cuda()
    print(next(model.parameters()).device)
    params = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(params, lr=1e-3, weight_decay=0.0001)

    # Loss function
    criterion = nn.BCELoss()
    lr_schedular = optim.lr_scheduler.StepLR(model_optim, step_size=5, gamma=0.1)

    num_epochs = 20

    for epoch in range(num_epochs):
        
        iterator = tqdm(dataloader, unit='img')
        model.train()
        for i, data in enumerate(iterator):
            iterator.set_description('Epoch %i/%i' % (epoch + 1, num_epochs))

            img = data['image']
            label = data['label']
            if USE_CUDA:
                img = img.cuda()
                label = label.cuda()

            model_optim.zero_grad()d
            
            outputs = model(img)
            loss = criterion(outputs, label)

            #epoch_loss += loss.item()

            loss.backward()
            model_optim.step()

            iterator.set_postfix(loss='{}'.format(loss.data))
        model.eval()
        accuracy=0
        with torch.no_grad():
            for data in testSet:
                img=data['image']
                label=data['label']
                if USE_CUDA:
                    img=img.cuda()
                output=model(img.unsqueeze(0)).cpu().numpy()
                label=label.numpy().astype(np.int64)
                pred=np.where(output>0.5,1,0).astype(np.int64)
                accuracy+=np.sum(np.equal(label,pred))/label.size
        print("The %i epoch, accuracy is %f"%(epoch,accuracy/len(testSet)))
        lr_schedular.step()


if __name__ == '__main__':
    main()

