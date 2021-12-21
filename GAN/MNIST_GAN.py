import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm

# https://arxiv.org/pdf/1406.2661.pdf

# hyperparameters
num_epochs = 100
latent_size = 64
hidden_size = 256
batch_size = 100
lr = 0.0002
beta_1= 0.5
beta_2 = 0.999

# device config
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# get dataset
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST(root='./data', train=True, transform=trans, download=True)

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
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True
)

# discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(28*28, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.network(input)
        return output

discriminator_model = Discriminator().to(device)
print("\nDiscriminator Model:")
print(summary(discriminator_model, (1, 28*28)))

# generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 28*28),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.network(input)
        return output

generator_model = Generator().to(device)
print("\nGenerator Model:")
print(summary(generator_model, (1, latent_size)))
print("\n")

# define loss function and optimizers
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(generator_model.parameters(), lr=lr)

d_losses = []
g_losses = []

# train
total_step = len(train_loader)
for epoch in range(num_epochs):
    epoch_g_loss = 0
    epoch_d_loss = 0
    for i, (images, _) in enumerate(tqdm(train_loader, desc="Epoch: {}/{}".format(epoch+1, num_epochs))):
        # load images and labels
        real_images = images.view(batch_size, -1).to(device)
        real_labels = torch.ones((batch_size, 1)).to(device)
        fake_labels = torch.zeros((batch_size, 1)).to(device)

        # train discriminator
        outputs = discriminator_model(real_images)
        d_loss_real = criterion(outputs, real_labels)

        latent_points = torch.rand(batch_size, latent_size).to(device)
        fake_images = generator_model(latent_points)
        outputs = discriminator_model(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)

        d_optimizer.zero_grad()
        d_loss_real.backward()
        d_loss_fake.backward()
        d_optimizer.step()

        epoch_d_loss += d_loss_real + d_loss_fake

        # train generator
        latent_points = torch.rand(batch_size, latent_size).to(device)
        fake_images = generator_model(latent_points)
        outputs = discriminator_model(fake_images)
        g_loss = criterion(outputs, real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        epoch_g_loss += g_loss

    d_losses.append(epoch_d_loss.item() / i)
    g_losses.append(epoch_g_loss.item() / i)

# plot losses
plt.figure()
plt.plot(d_losses, label='Discriminator loss')
plt.plot(g_losses, label='Generator Loss')
plt.title("Discriminator and Generator Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# plot sample generated images
latent_points = torch.rand(batch_size, latent_size).to(device)
fake_images = generator_model(latent_points)
fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
figure = plt.figure(figsize=(8, 8))
plt.title("Generated Images")
plt.axis("off")
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(fake_images), size=(1,)).item()
    img = fake_images.cpu().detach().numpy()[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
