import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor

def getFolderPath():
    fileStream = open("..\\path.txt", "r")
    return fileStream.read()

def dataUpload():
    path = getFolderPath()
    dataset = torch.utils.data.IterableDataset() 
    #trzeba skuonstruować klasę dziedziczącą po IterableDataset
    trainloader = DataLoader(dataset)
    return 0


def main():
    dataUpload()

main()