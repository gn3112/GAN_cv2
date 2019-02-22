import math
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from IPython.display import clear_output, display

# Testing git


class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.conv1 = nn.ConvTranspose2d(latent_size, 32, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=2, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1)

    def forward(self, z):
        z = z.view(-1, self.latent_size, 1, 1)
        x = F.relu(self.bn1(self.conv1(z)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return torch.sigmoid(self.conv4(x))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Takes input of 32x32x1
        # Skipping 2 pixels
        self.conv1 = nn.Conv2d(1, 8, 4, stride=2, padding=1, bias=False)
        #
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 4, stride=2, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        return torch.sigmoid(self.conv4(x)).view(-1)

        plt.axis('off')

def train(generator, discriminator, gen_optimiser, disc_optimiser, train_loader, batch_size, latent_size):
    for i, (real_x, _) in enumerate(train_loader):
        disc_optimiser.zero_grad()
        # Train discriminator to identify real data
        real_y = discriminator(real_x)
        real_loss = F.binary_cross_entropy(real_y, torch.ones_like(real_y))
        real_loss.backward()
        # Train discriminator to identify fake data
        noise = torch.randn(batch_size, latent_size)
        fake_x = generator(noise)
        fake_y = discriminator(fake_x.detach())
        fake_loss = F.binary_cross_entropy(fake_y, torch.zeros_like(fake_y))
        fake_loss.backward()
        disc_optimiser.step()

        gen_optimiser.zero_grad()
        # Train generator to fool discriminator on fake data
        fake_y = discriminator(fake_x)
        fake_loss = F.binary_cross_entropy(fake_y, torch.ones_like(fake_y))
        fake_loss.backward()
        gen_optimiser.step()


def sample(generator):
    generator.eval()
    black_bar = torch.zeros(3, 8 * 28, 10)
    with torch.no_grad():
        z_samples = torch.randn(64, 10)
        z_interp = torch.zeros(64, 10)
        samples = make_grid(generator(z_samples), padding=0)
        interps = make_grid(generator(z_interp), padding=0)
        plt.imshow(np.transpose(torch.cat([samples, black_bar, interps], 2).numpy(), [1, 2, 0]))
        pl.savefig("mnist.png")
        plt.show()
        clear_output(wait=True)
        display(plt.gcf())

def main():
    latent_size = 1
    batch_size = 64

    data_path = os.path.join(os.path.expanduser('~'), '.torch', 'datasets', 'mnist')
    train_data = datasets.MNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(data_path, train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    print(train_loader)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)
    ptin

    generator = Generator(latent_size)
    discriminator = Discriminator()
    gen_optimiser = optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    disc_optimiser = optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))

    for f in range(10):
        train(generator, discriminator, gen_optimiser, disc_optimiser, train_loader, batch_size, latent_size)
        sample(generator)
    clear_output(wait=True)


if __name__ == "__main__":
    main()
