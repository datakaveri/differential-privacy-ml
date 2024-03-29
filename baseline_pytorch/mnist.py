import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from mnist_datasets import MNISTDatasetTrain, MNISTDatasetTest

epochs = 100
batch_size = 600
lr = 0.05
momentum = 0.9

train_loader = torch.utils.data.DataLoader(MNISTDatasetTrain(), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(MNISTDatasetTest(), batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(60, 1000)
        self.fc2 = nn.Linear(1000, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    
model = Net()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

train_losses = []
test_losses = []

def train(epoch):
    model.train()
    epoch_loss = 0
    samples = 0
    batches = int(len(train_loader.dataset) / (batch_size*5))
    count = 0
    print('Training Epoch: {} ['.format(epoch) + '-'*batches+']', end='\r')
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, torch.flatten(target))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        samples += len(data)
        i = int(samples/(batch_size*5))
        count += 1
        print('Training Epoch: {} ['.format(epoch) +'='*(i)+ '-'*(batches-i)+'] ({}/{})'.format(samples, len(train_loader.dataset)), end='\r')
    print('Training Epoch: {} ['.format(epoch) + '='*batches+'] ({}/{})'.format(samples, len(train_loader.dataset)))
    print('Trained Epoch: {} with loss: {:.6f}'.format(
            epoch,
            epoch_loss / count
    ))
    train_losses.append(epoch_loss / count)
    torch.save(model.state_dict(), './saved/model.pth')
    torch.save(optimizer.state_dict(), './saved/optimizer.pth')
            
def test():
    model.eval()
    test_loss = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, torch.flatten(target)).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            count+=1
    test_loss /= count
    test_losses.append(test_loss)
    print('Test set metrics: \tAvg. loss: {:.4f}, \tAccuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
    
    
for epoch in range(1, epochs + 1):
    train(epoch)
    test()
    
fig = plt.figure()
plt.plot(np.arange(1, 101, 1), train_losses, color='blue')
plt.plot(np.arange(1, 101, 1), test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()