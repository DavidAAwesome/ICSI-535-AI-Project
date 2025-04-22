
# Author: Liam Lowry
# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.01
num_epochs = 35


# Device will determine whether to run the training on GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))

# to reformat images for modeling and some data augmentation
all_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = [0.491, 0.4822, 0.4465],
                                                          std = [0.2023, 0.1994, 0.2010])])

train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                             train=True,
                                             transform=all_transforms,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            transform=all_transforms,
                                            download=True)

# Instantiate loader objects to facilitate processing
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

class ConvNeuralNet(nn.Module):
# Determine what layers and their order in CNN object
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.convolution_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.convolution_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        self.convolution_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.convolution_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.fully_connected_layer1 = nn.Linear(1600,128)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.fully_connected_layer2 = nn.Linear(128,num_classes)

    def forward(self, x):
        out = self.convolution_layer1(x)
        out = self.convolution_layer2(out)
        out = self.bn1(out)
        out = self.Relu(out)
        out = self.max_pooling_layer(out)

        out = self.convolution_layer3(out)
        out = self.convolution_layer4(out)
        out = self.bn2(out)
        out = self.Relu(out)
        out = self.max_pooling_layer(out)

        out = out.reshape(out.size(0), -1)

        out = self.fully_connected_layer1(out)
        out = self.dropout(out)
        out = self.Relu(out)
        out = self.fully_connected_layer2(out)
        return out


model = ConvNeuralNet()
model = model.to(device)

# Set Loss function with criterion
criterion = nn.CrossEntropyLoss()

# Set optimizer with optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)



total_step = len(train_loader)
model.train()
# training
for epoch in range(num_epochs):
# data loaded in batches using train_loader
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# testing
model.eval()
with torch.no_grad():
    correctTrain = 0
    totalTrain = 0
    correctTest = 0
    totalTest = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        totalTrain += labels.size(0)
        correctTrain += (predicted == labels).sum().item()
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        totalTest += labels.size(0)
        correctTest += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correctTrain / totalTrain ))
    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correctTest / totalTest))