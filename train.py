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
from model.NestedUnet import NestedUNet
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

    model = NestedUNet(1,1)
    if USE_CUDA:
        model=model.cuda()
    print(next(model.parameters()).device)
    params = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(params, lr=1e-3)

    # Loss function
    lr_schedular = optim.lr_scheduler.StepLR(model_optim, step_size=20, gamma=0.5)

    num_epochs = 60

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
            model_optim.zero_grad()
            BCEweight=torch.where(label==0,2,1)
            criterion = nn.BCEWithLogitsLoss(weight=BCEweight)
            outputs = model(img)
            loss = criterion(outputs, label)


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
                    label=label.cuda()
                output=model(img.unsqueeze(0))
                BCEweight=torch.where(label[0]==0,2,1).squeeze()
                criterion = nn.BCEWithLogitsLoss(weight=BCEweight)
                loss = criterion(output.squeeze(), label[0].squeeze())
                accuracy+=loss.item()
        print("The %i epoch, loss is %f"%(epoch+1,accuracy/len(testSet)))
        torch.save(model.state_dict(), './save_model/NestedUnet_'+str(epoch+1)+'epoch_'+str(accuracy/len(testSet))+'loss.pth')
        lr_schedular.step()


if __name__ == '__main__':
    main()

