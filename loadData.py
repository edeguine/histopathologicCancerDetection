import PIL
from PIL import Image
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class ImageDataset(Dataset):
    def __init__(self, imageFilenames, labels, flipHorizontal=False, flipVertical=False, meanNorm = True, stdNorm = True):
        self.imageFilenames = imageFilenames
        self.labels = labels 
        self.flipHorizontal = flipHorizontal
        self.flipVertical = flipVertical
        self.meanNorm = meanNorm
        self.stdNorm = stdNorm

        print('flipV', flipVertical)

    def __len__(self):
        return len(self.imageFilenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x = Image.open(self.imageFilenames[idx])
        if self.flipHorizontal and random.randint(0, 1) == 1:
            x = flipImage(x, 'horizontal')

        if self.flipVertical and random.randint(0, 1) == 1:
            x = flipImage(x, 'vertical')

        if self.meanNorm:
            x = x - np.mean(x)

        if self.stdNorm:
            x = x / np.std(x)

        x = reorg(x)

        sample = (x, self.labels[idx])
        return sample

class TestDataset(Dataset):
    def __init__(self, imageFilenames, shortNames, meanNorm=False):
        self.imageFilenames = imageFilenames
        self.shortNames = shortNames
        self,meanNorm = meanNorm

    def __len__(self):
        return len(self.imageFilenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x = Image.open(self.imageFilenames[idx])
        if self.meanNorm:
            x = x - np.mean(x)

        x = reorg(x)

        sample = (x, self.shortNames[idx])

        return sample

class TestClassification:
    def __init__(self, version):
        self.csv = open('classification/' + version + '_labels.csv', 'w')

    def write(self, filename, label):
        filename = str(filename[0])
        line = ','.join([filename, str(label)]) + '\n'
        self.csv.write(line)

    def close(self):
        self.csv.close()

def flipImage(im, direction):
    if direction == 'horizontal':
        im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    elif direction == 'vertical':
        im.transpose(PIL.Image.FLIP_TOP_BOTTOM)

    return im

def makeIm(filename):
    im = Image.open(filename)
    imnp = np.array(im)
    channels = [imnp[:, :, i] for i in range(3)]
    channels = np.stack(channels, axis=0)
    channels = channels.astype('float')
    channels = channels / 128.0 - 1
    
    return channels

def reorg(im):
    imnp = np.array(im)
    channels = [imnp[:, :, i] for i in range(3)]
    channels = np.stack(channels, axis=0)
    channels = channels.astype('float')
    channels = channels / 128.0 - 1
    
    return channels


def loadDataset(version='train', **kwargs):

    root = "/home/ubuntu/histoCancerKaggle/data/"
    ref = open(root + version + '_labels.csv').readlines()[1:]

    imageFilenames = []
    labels = []
    ct = 0
    labelsCt = {0: 0, 1: 0}
    for line in ref:
        name, label = line.split(',')
        if 'synth' in version:
            name = version + '/' + name + '.png'
        else:
            name = root + version + '/' + name + '.tif'
        try:
            a = open(name)
            imageFilenames.append(name)
            labels.append(int(label))
            labelsCt[int(label)] += 1
            ct += 1
        except FileNotFoundError:
            pass

    print(f"{ct} images found in dir")
    print(f"0: {labelsCt[0]} 1: {labelsCt[1]}")

    dataset = ImageDataset(imageFilenames, labels, **kwargs)

    return dataset

def loadTestSet(version='train'):

    root = "/home/ubuntu/histoCancerKaggle/data/"
    ref = open(root + version + '_files.csv').readlines()[1:]

    imageFilenames = []
    shortNames = []
    ct = 0
    for line in ref:
        name = line.rstrip('\n')
        shortName = name
        if 'synth' in version:
            name = version + '/' + name + '.png'
        else:
            name = root + version + '/' + name + '.tif'
        try:
            a = open(name)
            imageFilenames.append(name)
            shortNames.append(shortName)
            ct += 1
        except FileNotFoundError:
            pass

    print(f"{ct} images found in dir")

    dataset = TestDataset(imageFilenames, shortNames)

    return dataset


def examine(image):
    print(f"Shape {image.shape}")
    print(image[:,:,:])
    print(f"min: {np.min(image)} max; {np.max(image)}")

def load(filename):
    makeIm(filename)

def test():
    #filename = "train/fffeeb1297fd4e26f247af648a2a6f942dfa2e9d.tif"
    #load(filename)

    #dataset = loadDataset('synthtrain')
    #im, label = dataset[0]
    #examine(im)

    dataset = loadDataset('mediumtrain', flipHorizontal=True, flipVertical=True, meanNorm=True, stdNorm=True)
    im, label = dataset[0]
    examine(im)

    dataset = loadDataset('mediumtest')
    im, label = dataset[0]
    examine(im)
    
    dataset = loadTestSet('test')
    im, label = dataset[0]
    print(label)
    examine(im)


if __name__ == "__main__":
    test()
