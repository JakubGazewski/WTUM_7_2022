import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor

from fishdataset import FishDatasetLoader

def getMasterFolderPath():
    fileStream = open("..\\path.txt", "r")
    return fileStream.read()

def dataUpload():
    master_folder_path = getMasterFolderPath()
    train_images_path = master_folder_path + "\\train"

    fish_loader = FishDatasetLoader(train_images_path)
    fish_loader.doWork()


def main():
    dataUpload()

main()