# Imports
import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader # Gives easier dataset management and creates mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms # Transformations we can perform on our dataset


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()

        # First convolutional layer
        # Takes in input with 'in_channels' (e.g., 1 for grayscale images)
        # Outputs 8 feature maps (8x28x28)
        # Uses a 3x3 kernel with stride of 1 and padding of 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))

        # Max pooling layer
        # Reduces the size of the feature maps by half
        # Uses a 2x2 kernel with stride of 2
        # Output: 8x14x14
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Second convolutional layer
        # Takes in 8 feature maps from previous layer
        # Outputs 16 feature maps (16x14x14)
        # Uses a 3x3 kernel with stride of 1 and padding of 1
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Fully connected layer
        # Takes the flattened feature maps from conv2 and pool layers
        # Outputs 'num_classes' (e.g., 10 for digit classification) (16x7x7)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        # Apply first convolutional layer followed by ReLU activation
        x = F.relu(self.conv1(x))

        # Apply max pooling
        x = self.pool(x)

        # Apply second convolutional layer followed by ReLU activation
        x = F.relu(self.conv2(x))

        # Apply max pooling again
        x = self.pool(x)

        # Flatten the tensor from 4D (batch_size, channels, height, width) to 2D (batch_size, flattened_features)
        x = x.reshape(x.shape[0], -1)

        # Apply fully connected layer
        x = self.fc1(x)

        return x


model = CNN()
x = torch.randn(64, 1, 28, 28)
print(model(x).shape)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels=1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2
load_model = True

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_model(torch.load("my_checkpoint.path.tar"))


# Train Network
for epoch in range(num_epochs):
    losses = []

    if epoch == 2:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


# Check accuracy on training & test to see how good our model

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} correct with accuracy {float(num_correct) / float(num_samples) * 100:.2f}%')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)





