from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

train_label_path = './data/train_label/'
train_img_path = './data/train_img/'


class BasicDataSet(Dataset):
    def __init__(self, img_dir, label_dir, scale=1):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.scale =1
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.dataIdx = [splitext(file)[0] for file in listdir(img_dir)]
        logging.info(f'Creating dataset with {len(self.dataIdx)} examples')

    def __len__(self):
        return len(self.dataIdx)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale*w), int(scale*h)
        assert newH > 0 and newW > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, item):
        idx = self.dataIdx[item]
        label_file = glob(self.label_dir + idx + '.png')
        img_file = glob(self.img_dir + idx + '.*')

        assert len(label_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {label_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        label = Image.open(label_file[0])
        img = Image.open(img_file[0])

        assert label.size == img.size, f'Image and mask {idx} should have the same size, but are {img.size} and {label.size}'

        img = self.preprocess(img, self.scale)
        label = self.preprocess(label, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'label': torch.from_numpy(label).type(torch.FloatTensor)
        }


if __name__ == '__main__':
    pass
