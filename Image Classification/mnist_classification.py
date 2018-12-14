import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,20,3)
        self.conv3 = nn.Conv2d(20,30,3)
        self.conv3drop = nn.Dropout2d()
        self.fc1 = nn.Linear(14520,50)
        self.fc2 = nn.Linear(50,10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu( self.conv3drop(x) )
        x = x.view(-1,14520)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x,dim=1)

def train(model,device,train_loader,optimizer,criterion,epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()

        output = model(data) # [1,10]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train epoch: ',epoch,' loss: {:.4f}'.format(loss.item()),
            ' batch done {}/{}'.format(batch_idx * len(data), len(train_loader.dataset)))

def test(model,device,test_loader,criterion,epoch,testloss_lis):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output,target).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    testloss_lis.append(test_loss)

    print('\n Avg loss: ',test_loss,' Accuracy {}/{}'.format(correct,len(test_loader.dataset)),
    ' ({:.0f}%)'.format(100 * correct / len(test_loader.dataset)))

        
def main():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            'data',download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
            ])))
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data',train=False,
        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))            
            ])))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ConvNet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    
    testloss_lis = []
    epoch_lis = [i for i in range(1,5)]

    for epoch in range(1, 5):
        train(model,device,train_loader,optimizer,criterion,epoch)
        test(model,device,test_loader,criterion,epoch,testloss_lis)
    
    plt.plot(epoch_lis, testloss_lis)
    plt.xlabel('epochs(k)')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig('test.jpg')
    
if __name__ == '__main__':
    main()
        