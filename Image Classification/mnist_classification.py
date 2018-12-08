import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import datasets, transforms

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,20,3)
        self.conv3 = nn.Conv2d(20,30,3)
        self.conv3drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250,50)
        self.fc2 = nn.Linear(50,10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu( self.conv3drop(self.conv3( self.conv2(x) ) ) )
        x = x.view(-1,250)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x,dim=1)

def train(model,device,train_loader,optimizer,criterion,epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train epoch: ',epoch,' loss:',loss.item())

        


def main():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.5,0.5)
            ])))
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data',train=False,
        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.5,0.5)            
            ])))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ConvNet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    for epoch in range(1, 10):
        train(model,device,train_loader,optimizer,criterion,epoch)

if __name__ == '__main__':
    main()
        