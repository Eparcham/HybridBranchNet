import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import timeit as benchmark
import time


class ParallelNet(nn.Module):
    def __init__(self):
        super(ParallelNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_2 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.pool1_1(x1)
        x1 = self.conv2_1(x1)
        x1 = self.pool2_1(x1)

        x2 = self.conv1_2(x)
        x2 = self.pool1_2(x2)
        x2 = self.conv2_2(x2)
        x2 = self.pool2_2(x2)

        x = torch.add(x1, x2)
        x = self.flatten(x)
        x = self.fc1(x)
        out = self.out(x)
        return out
class SeriesNet(nn.Module):
    def __init__(self):
        super(SeriesNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        out = self.out(x)
        return out


parallel_net = ParallelNet()
series_net = SeriesNet()
# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parallel_net.to(device)
series_net.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"count_parameters parallel_net: {count_parameters(parallel_net)} count_parameters series_net: {count_parameters(series_net)}")
# Define transformations for dataset
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=8)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer_parallel = torch.optim.Adam(parallel_net.parameters(), lr=lr)
optimizer_series = torch.optim.Adam(series_net.parameters(), lr=lr)

# Train the ParallelNet and SeriesNet
num_epochs = 100
for epoch in range(num_epochs):
    # Train ParallelNet
    parallel_net.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_parallel.zero_grad()

        outputs = parallel_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_parallel.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print('[Epoch %d, Batch %d] ParallelNet loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

    # Train SeriesNet
    series_net.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_series.zero_grad()

        outputs = series_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_series.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print('[Epoch %d, Batch %d] SeriesNet loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

    # Evaluate both networks on test set
    parallel_net.eval()
    series_net.eval()
    correct_parallel, correct_series = 0, 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs_parallel = parallel_net(images)
            _, predicted_parallel = torch.max(outputs_parallel.data, 1)
            correct_parallel += (predicted_parallel == labels).sum().item()

            outputs_series = series_net(images)
            _, predicted_series = torch.max(outputs_series.data, 1)
            correct_series += (predicted_series == labels).sum().item()

            total += labels.size(0)

    print('[Epoch %d] ParallelNet accuracy on test set is %d %%' % (epoch + 1, 100 * correct_parallel / total))
    print('[Epoch %d] SeriesNet accuracy on test set is %d %%' % (epoch + 1, 100 * correct_series / total))
    print('--------------------------------------------------------------')


# Measure inference time for ParallelNet
parallel_net.eval()
start_time = time.time()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = parallel_net(images)
inference_time_parallel = time.time() - start_time
print("ParallelNet inference time: {:.4f} seconds".format(inference_time_parallel))

# Measure inference time for SeriesNet
series_net.eval()
start_time = time.time()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = series_net(images)
inference_time_series = time.time() - start_time
print("SeriesNet inference time: {:.4f} seconds".format(inference_time_series))

