import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt 
import time 
import os 
import copy 

train_loader = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_loader = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'hymenoptera_data'
train_image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_loader) 
test_image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), test_loader)

train_image_dataloader = torch.utils.data.DataLoader(train_image_dataset, 
                            batch_size=4, shuffle=True, num_workers=0)
test_image_dataloader = torch.utils.data.DataLoader(test_image_dataset, 
                            batch_size=4, shuffle=True, num_workers=0)
dataset_sizes = {'train': len(train_image_dataset), 'val': len(test_image_dataset)}
class_names = train_image_dataset.classes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1,2,0)) # inp = [3, 228, 906]. after transpose inp = [228, 906, 3]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp) # after clipping inp has maxm = 1.0 and minm 0.0. Shape of inp remains same
    if title is not None:
        plt.title(title)
    plt.show()

inputs, classes = next(iter(train_image_dataloader)) # input = [4, 3, 224, 224]

out = torchvision.utils.make_grid(inputs) # out = [3, 228, 906]

# imshow(out, title=[class_names[x] for x in classes])

def train_model(model, optimizer, criterion):
    start_time = time.time()
    epochs = 1
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_image_dataloader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # make all the gradients zero

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval() # testing mode. so eval() mode on
                
                with torch.no_grad():
                    for inputs, targets in test_image_dataloader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, targets)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        prediction = torch.exp(logps)
                        top_p, top_class = prediction.topk(1, dim=1)
                        equals = top_class = targets.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(test_image_dataset):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_image_dataset):.3f}")
                running_loss = 0



def main():
    model = models.densenet121(pretrained=True)
    num_fts = 1024

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(num_fts,2),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

    model.to(device)

    train_model(model, optimizer, criterion)

if __name__ == "__main__":
    main()







    








