import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor

def getFolderPath():
    fileStream = open("..\\path.txt", "r")
    print(fileStream.read())

def main():
    getFolderPath()

main()