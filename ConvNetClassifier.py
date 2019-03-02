# Name: Peng Cheng
# UIN: 674792652
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Configuration area
#############################
epochs = 50
batch_size = 50
learning_rate = 0.001
num_workers = 4
load_pretrained_model = True
pretrained_epoch = 25
#############################
use_cuda = True
device = torch.cuda.device("cuda" if use_cuda else "cpu")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainset_loader = torch.utils.data.DataLoader(cifar_trainset, batch_size=batch_size,shuffle=True, num_workers=num_workers)

cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testset_loader = torch.utils.data.DataLoader(cifar_testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(in_features=8*8*256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.Dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #32*32*48
        x = F.relu(self.conv2(x)) #32*32*96
        x = self.pool(x) #16*16*96
        x = self.Dropout(x)
        x = F.relu(self.conv3(x)) #16*16*192
        x = F.relu(self.conv4(x)) #16*16*256
        x = self.pool(x) # 8*8*256
        x = self.Dropout(x)
        x = x.view(-1, 8*8*256) # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x
# Define loss function and optimizer. We employ cross-entropy and Adam
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# define the test accuracy function
def test_accuracy(net, testset_loader, epoch):
    # Test the model
    net.eval()
    correct = 0
    total = 0
    for data in testset_loader:
        images, labels = data
        images, labels = Variable(images).cuda(), labels.cuda()
        output = net(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network after epoch '+str(epoch+1)+' is: ' + str(100 * correct / total))

#We save the model after every 5 epochs
def save_model(net, epoch):
    filename = "model_with_epoch" + str(epoch+1) + ".pth"
    torch.save(net.state_dict(),filename)

#Train the neural network and test for accuracy after every 5 epochs
#Depend on whether we need to load the pretrained model
if load_pretrained_model == False:
    net = ConvNet()
    net.cuda()
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(epochs):
        running_loss = 0.0
        if epoch <= 10:
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        elif epoch > 10 and epoch <= 25:
            optimizer = optim.Adam(net.parameters(), lr=(learning_rate)/10)
        else:
            optimizer = optim.Adam(net.parameters(), lr=(learning_rate)/50)       
        for i, data in enumerate(trainset_loader):
            input_data, labels = data # data is a list of 2, the first element is 4*3*32*32 (4 images) the second element is a list of 4 (classes)
            input_data, labels = Variable(input_data).cuda(),Variable(labels).cuda()
            optimizer.zero_grad() # every time reset the parameter gradients to zero
            # forward backward optimize
            output = net(input_data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            # print the loss
            running_loss += loss.data[0]
        # print the loss after every epoch
        print('loss in epoch ' + str(epoch + 1) + ': ' + str(running_loss / 50000))    
        if (epoch + 1)%5 == 0:
            # Test for accuracy after every 5 epochs
            test_accuracy(net, testset_loader, epoch)
            # Save model after every 5 epochs
            save_model(net, epoch)
        elif epoch == epochs - 1:
            test_accuracy(net, testset_loader, epoch)
            save_model(net, epoch)
else:
    net = ConvNet()
    net.load_state_dict(torch.load("model_with_epoch" + str(pretrained_epoch) + ".pth"))
    net.cuda()
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(pretrained_epoch, epochs):
        running_loss = 0.0
        if epoch <= 10:
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        elif epoch > 10 and epoch <= 25:
            optimizer = optim.Adam(net.parameters(), lr=(learning_rate)/10)
        else:
            optimizer = optim.Adam(net.parameters(), lr=(learning_rate)/50)
        for i, data in enumerate(trainset_loader):
            input_data, labels = data # data is a list of 2, the first element is 4*3*32*32 (4 images) the second element is a list of 4 (classes)
            input_data, labels = Variable(input_data).cuda(),Variable(labels).cuda()
            optimizer.zero_grad() # every time reset the parameter gradients to zero
            # forward backward optimize
            output = net(input_data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            # print the loss
            running_loss += loss.data[0]
        # print the loss after every epoch
        print('loss in epoch ' + str(epoch + 1) + ': ' + str(running_loss / 50000))
        if (epoch + 1)%5 == 0:
            # Test for accuracy after every 5 epochs
            test_accuracy(net, testset_loader, epoch)
            # Save model after every 5 epochs
            save_model(net, epoch)
        elif epoch == epochs - 1:
            test_accuracy(net, testset_loader, epoch)
            save_model(net, epoch)        

print("Training and Testing Completed!")
