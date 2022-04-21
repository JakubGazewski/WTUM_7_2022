import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, IterableDataset, DataLoader
from matplotlib import pyplot as plt
import numpy as np
from torchvision import datasets, transforms, models

class FishDatasetLoader:
    def __init__(self, train_images_path) -> None:
        self.train_images_path = train_images_path

    def getData(self):
        print("creating loader...")
        valid_size = 0.2

        _transforms = transforms.Compose([
            transforms.RandomEqualize(1),
            transforms.Resize((256, 256)), 
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.2, 0.2, 0.2]
            ),
            ])

        train_data = datasets.ImageFolder(self.train_images_path, transform=_transforms)
        test_data = datasets.ImageFolder(self.train_images_path, transform=_transforms)

        #getting test images randomly selected from train ones
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)
        from torch.utils.data.sampler import SubsetRandomSampler
        train_idx, test_idx = indices[split:], indices[:split]
        
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        trainloader = torch.utils.data.DataLoader(train_data,
                    sampler=train_sampler, batch_size=32)
        testloader = torch.utils.data.DataLoader(test_data,
                    sampler=test_sampler, batch_size=32)

        return trainloader, testloader

    
    def getModel(self):
        print("creating model...")
        print(torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet50(pretrained=True)
        # print(model)

        for param in model.parameters():
            param.requires_grad = False
    
        model.fc = nn.Sequential(nn.Linear(2048, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 10),
                                        nn.LogSoftmax(dim=1))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
        model.to(device)
        return model, device, optimizer, criterion
    
    # for now there is no testset (testset = trainset)
    def trainModel(self, model, trainloader, testloader, device, optimizer, criterion):
        print("training...")

        epochs = 40
        steps = 0
        running_loss = 0
        print_every = 100
        train_losses, test_losses = [], []
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % print_every == 0:

                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in testloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            test_loss += batch_loss.item()
                            
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    train_losses.append(running_loss/len(trainloader))
                    test_losses.append(test_loss/len(testloader))                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Test loss: {test_loss/len(testloader):.3f}.. "
                        f"Test accuracy: {accuracy/len(testloader):.3f}")
                    running_loss = 0
                    model.train()

        torch.save(model, 'aerialmodel.pth')
        return train_losses, test_losses
    
    def check_work(self, train_losses, test_losses):
        plt.plot(train_losses, label='Training loss')
        plt.plot(test_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()

    def doWork(self):
        trainloader, testloader = self.getData()
        model, device, optimizer, criterion = self.getModel()
        train_losses, test_losses = self.trainModel(model, trainloader, testloader, device, optimizer, criterion)
        self.check_work(train_losses, test_losses)