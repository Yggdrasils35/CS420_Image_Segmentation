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
from model.NestedUnet import NestedUNet

from sklearn.metrics import accuracy_score
from PIL import Image


test_img='./data/test_img/'
test_label='./data/test_label/'
model_path='save_model/NestedUnet_23epoch_0.2758275091648102loss.pth'
def main():
    if torch.cuda.is_available():
        USE_CUDA = True
        print('Using GPU, %i devices.' % torch.cuda.device_count())
    else:
        USE_CUDA = False

    
    testSet = BasicDataSet(test_img,test_label,1)

    model = NestedUNet(1,1)
    model.load_state_dict(torch.load(model_path))
    if USE_CUDA:
        model=model.cuda()
    
    model.eval()
    accuracy=0
    i=0
    with torch.no_grad():
        for data in testSet:
            img=data['image']
            label=data['label']
            if USE_CUDA:
                img=img.cuda()
            output=model(img.unsqueeze(0))
            output=torch.sigmoid(output)
            #print(img)

            #pred=torch.where(output>0.6,1,0)
            #print(pred.sum())
            output=torch.where(output>0.5,1,0)
            pred=(output*255).cpu()
            #print(pred.squeeze())
            #print(pred)
            pred_img=Image.fromarray(pred.squeeze().numpy().astype(np.int8))
            pred_img.convert('L').save('./predict_img/NestedUnet_'+str(i)+'.png')
            label_img=Image.fromarray(np.int8(255*label[0].cpu()))
            label_img.convert('L').save('./predict_img/Label_'+str(i)+'.png')
            i+=1


if __name__ == '__main__':
    main()