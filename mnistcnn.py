import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

batch_size = 64
num_classes = 10
learning_rate = 0.01
num_epochs = 10  # MNIST converges faster
classes = [str(i) for i in range(10)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Transforms for MNIST (no color channels, just resizing and normalization)
all_transforms = transforms.Compose([
    transforms.Resize((32, 32)),  # resize so we can reuse your CNN structure
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root="./data",
                                           train=True,
                                           transform=all_transforms,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root="./data",
                                          train=False,
                                          transform=all_transforms,
                                          download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Updated CNN for MNIST (input channels = 1)
class ConvNeuralNet(nn.Module):
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.convolution_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.convolution_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        self.convolution_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.convolution_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        # NOTE: Adjusted input size for fully connected layer.
        # Input image is now 32x32, after 2 max poolings → around 5x5 feature maps
        self.fully_connected_layer1 = nn.Linear(64 * 5 * 5, 128)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.fully_connected_layer2 = nn.Linear(128, num_classes)

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

        out = out.view(out.size(0), -1)  # flatten

        out = self.fully_connected_layer1(out)
        out = self.dropout(out)
        out = self.Relu(out)
        out = self.fully_connected_layer2(out)
        return out

model = ConvNeuralNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

def train():
    total_step = len(train_loader)
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'mnist_model_weights.pth')

def test():
    model.load_state_dict(torch.load('mnist_model_weights.pth'))
    model.eval()
    correctTrain, totalTrain = 0, 0
    correctTest, totalTest = 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            totalTrain += labels.size(0)
            correctTrain += (predicted == labels).sum().item()

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            totalTest += labels.size(0)
            correctTest += (predicted == labels).sum().item()

    print(f'Train Accuracy: {100 * correctTrain / totalTrain:.2f}%')
    print(f'Test Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%')

    print(classification_report(y_true, y_pred, target_names=classes))

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

train()
test()
