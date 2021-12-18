import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

# device config
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# get dataset
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST(root='./data', train=True, transform=trans, download=True)
test_set = datasets.MNIST(root='./data', train=False, transform=trans, download=True)

# visualize data
def visualize_data():
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_set), size=(1,)).item()
        img, label = train_set[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

# create data loaders
batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print("Train size: {}".format(train_loader.dataset.data.size()))
print("Test size: {}".format(test_loader.dataset.data.size()))

# discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Flatten())
        self.fc = nn.Sequential(
            nn.Linear(7*7*64, 1),
            nn.Sigmoid())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        return out

discriminator_model = Discriminator().to(device)
# print(summary(discriminator_model, (1, 28, 28)))

# loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=0.0002)

# train the discriminator model on half fake, half real data
total_step = len(train_loader)
num_epochs = 1
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if i % 2 == 0:
            images = torch.rand(100, 1, 28, 28)
            labels = torch.zeros((100,1))
        else:
            labels = torch.ones((100,1))

        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = discriminator_model(images)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
