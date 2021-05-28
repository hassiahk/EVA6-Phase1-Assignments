#%%
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
# %%
class Net(nn.Module):
    '''
    GOAL : get 99.5% accuracy with <20K Params

    Current: 6.3M - 26 MB

    1. Configuring arch with convolution/transition block after getting 7x7 GRF
    2. Pointwise near output to remove linear layer ?
    2. Try depthwise + pointwise convolution

    '''
    def __init__(self):
        super(Net, self).__init__()                                            # Input      Output      LRF     GRF
        
        #! convolution block
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)                            # 28x28x1    28x28x32    3x3     3x3
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(0.05)


        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)                           # 28x28x32   28x28x32    3x3     5x5
        self.bn2 = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout(0.05)
#        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)                           # 28x28x32   28x28x32    3x3     7x7
        

        #! transition block
        self.conv_point1 = nn.Conv2d(in_channels= 32,out_channels= 16, kernel_size =1,padding=1)                        # 28x28x32   28x28x16    3x3     7x7
        self.pool1 = nn.MaxPool2d(2, 2)                                        # 28x28x16   14x14x16            14x14 ?

        
        #! convolution block
        self.conv4 = nn.Conv2d(16, 32, 3)                                       # 14x14x16   12x12x32   3x3     16x16
#        self.bn3 = nn.BatchNorm2d(32)

#        self.conv5 = nn.Conv2d(32, 32, 3)                                       # 12x12x32  10x10x32    3x3     18x18


        #x = F.adaptive_avg_pool2d(x, (1, 1))
    
        # self.pool2 = nn.MaxPool2d(2, 2)                                        # 14x14x256    7x7x256     3x3     28x28
        # self.conv5 = nn.Conv2d(256, 512, 3)                                    # 7x7x256    5x5x512     3x3     30x30
        # self.conv6 = nn.Conv2d(512, 1024, 3)                                   # 5x5x512    3x3x1024    3x3     32x32
        # self.conv7 = nn.Conv2d(1024, 10, 3)                                   # 3x3x1024    1x1x10      3x3     34x34




        self.fc1 = nn.Linear(32,10)
        #self.fc2 = nn.Linear(1000,10)

    def forward(self, x):
#        print(x.shape)
        x = self.pool1(F.relu(self.conv_point1(self.drop2(self.bn2(F.relu(self.conv2(self.drop1(self.bn1(F.relu(self.conv1(x)))))))))))
        x = F.relu(self.conv4(x))
#        print('before GAP',x.shape)        
        x = F.adaptive_avg_pool2d(x, (1, 1)) #  10x10x32  -> 1x1x32 (actually 11x11x32 before)
#        print('After GAP',x.shape)       

        x = torch.flatten(x, 1)

#        print('After flatten',x.shape)

        x = self.fc1(x) #1x1x32 -> 1x1x10


        # x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        # x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # #x = F.relu(self.conv7(x))
        # x = self.conv7(x)
        #print('x after conv7 \t',x)
        #print('x after conv7 \t',x.shape)
        
        #added FC
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)

        #print('x after linear \t',x.shape)
        x = x.view(-1, 10)
        #print('x after view \t',x)
        #print('x after view \t',x.shape)
        return F.log_softmax(x)

# %%
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
# %%


torch.manual_seed(1)
batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False,
                    transform=transforms.Compose([
#                        transforms.ToPILImage(),
                        transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2),
#                        transforms.RandomRotation(30),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

# %%
from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
    correct = 0
    model.train()
    pbar = tqdm(train_loader)
    per_batch_loss = []

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        pred = output.argmax(dim=1, keepdim=True)          
        correct += pred.eq(target.view_as(pred)).sum().item()
        #pred.eq will give mask of 1,0 where the output's match

        loss = F.nll_loss(output, target)
        per_batch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
#        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')

    
    per_epoch_loss = sum(per_batch_loss)/len(per_batch_loss)
    per_epoch_accuracy = 100.*( correct / len(train_loader.dataset))
    
    return per_epoch_loss, per_epoch_accuracy

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


    return test_loss , 100. * correct / len(test_loader.dataset)


# %%
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_train = []
accuracy_train = []
loss_test = []
accuracy_test = []
epochs = 20
for epoch in range(0, epochs):
    
    loss_train_epoch,accuracy_train_epoch = train(model, device, train_loader, optimizer, epoch)
    loss_test_epoch, accuracy_test_epoch = test(model, device, test_loader)
    
    
    loss_train.append(loss_train_epoch)
    accuracy_train.append(accuracy_train_epoch)

    loss_test.append(loss_test_epoch)
    accuracy_test.append(accuracy_test_epoch)


    test(model, device, test_loader)

def plot_loss(train_loss, test_loss):
    plt.figure(figsize=(8,6))
    plt.plot(np.linspace(1, epochs, epochs).astype(int),loss_train,'-x')
    plt.plot(np.linspace(1, epochs, epochs).astype(int),loss_test,'-x')
    plt.legend(['train_loss', 'test_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')


def plot_accuracy(train_acc, test_acc):
    plt.figure(figsize=(8,6))
    plt.plot(np.linspace(1, epochs, epochs).astype(int),train_acc,'-x')
    plt.plot(np.linspace(1, epochs, epochs).astype(int),test_acc,'-x')
    plt.legend(['train_acc', 'test_acc'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')

plot_loss(loss_train,loss_test)
plot_accuracy(accuracy_train,accuracy_test)

print(accuracy_test)
# %%




# %%
