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
from torchvision.utils import make_grid, save_image
from IPython.display import clear_output, display

class Generator(nn.Module):
    def __init__(self,latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.conv1 = nn.ConvTranspose2d((self.latent_size+10), 256, 8, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 1, 4, stride=2, padding=0)

    def forward(self, z, y_label):
        z = z.view(-1, self.latent_size, 1, 1).detach()
        z = torch.cat((z,y_label),1).requires_grad_(True)
        print(z.size())
        x = F.relu(self.bn1(self.conv1(z)))
        print(x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        print(x.size())
        return torch.tanh(self.conv3(x))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(11, 128, 5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 1, 7, stride=1, padding=1)

    def forward(self, x, y_fill):
        print(x.size())
        print(y_fill.size())
        x = torch.cat((x,y_fill),1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        return torch.sigmoid(self.conv3(x)).view(-1)
        plt.axis('off')


def train(generator, discriminator, gen_optimiser, disc_optimiser, train_loader, batch_size, latent_size, ep):
    loss_g = []
    loss_d = []
    for batch_idx, (real_x, label) in enumerate(train_loader):


        y_label = torch.zeros(batch_size,10)
        y_label[:,label] = 1
        y_label = y_label.view(-1,10,1,1).float()

        y_fill = y_label * torch.ones(batch_size,10,28,28)
        y_fill = y_fill.view(-1,10,28,28).float()


        disc_optimiser.zero_grad()
        # Train discriminator to identify real data
        real_y = discriminator(real_x,y_fill)
        real_loss = F.binary_cross_entropy(real_y, torch.ones_like(real_y))
        real_loss.backward()

        # Train discriminator to identify fake data
        noise = torch.randn(batch_size, latent_size)
        fake_x = generator(noise,y_label)
        fake_y = discriminator(fake_x.detach(),y_fill)
        fake_loss = F.binary_cross_entropy(fake_y, torch.zeros_like(fake_y))
        fake_loss.backward()
        loss = (fake_loss + real_loss).detach().numpy()
        loss_d = np.append(loss_d, loss)
        disc_optimiser.step()
        gen_optimiser.zero_grad()

        # Train generator to fool discriminator on fake data
        fake_y = discriminator(fake_x,y_fill)
        fake_loss = F.binary_cross_entropy(fake_y, torch.ones_like(fake_y))
        fake_loss.backward()
        gen_optimiser.step()
        loss = fake_loss.detach().numpy()
        loss_g = np.append(loss_g, loss)

        if batch_idx % 50 == 0 and batch_idx !=0:
            print("---- Iteration: " + str(batch_idx) + " in Epoch: " + str(ep+1) + " ----")
            print("---- Loss Generator: " + str(np.around(loss_g[-1],decimals=2)) + " Loss Discriminator: " + str(np.around(loss_d[-1],decimals=2)) + "----")

    return loss_d, loss_g

def sample(generator):
    generator.eval()
    black_bar = torch.zeros(3, 8 * 28, 100)
    with torch.no_grad():
        z_samples = torch.randn(60, 100)
        z_interp = torch.zeros(60, 100)

        label = torch.tensor(0).repeat(6)
        for digits in range(9):
            label = torch.cat((label,torch.tensor(digits).repeat(6)),dim=0)
        label = label.float()


        samples = make_grid(generator(z_samples,label), padding=0)
        interps = make_grid(generator(z_interp,label), padding=0)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(np.transpose(torch.cat([samples, black_bar, interps], 2).numpy(), [1, 2, 0]))
        plt.imshow(np.transpose(torch.cat([samples, black_bar, interps], 2).numpy(), [1, 2, 0]))
        plt.show()
        # clear_output(wait=True)
        # display(plt.gcf())
        return fig

def main():
    plt.interactive(True)
    latent_size = 100
    batch_size = 64
    epoch = 10

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_path = os.path.join(os.path.expanduser('~'), '.torch', 'datasets', 'mnist')
    train_data = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_path, train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

    generator = Generator(latent_size)
    discriminator = Discriminator()
    gen_optimiser = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    disc_optimiser = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    plt.axis('off')
    loss_d_log = []
    loss_g_log = []
    for f in range(epoch):
        loss_d, loss_g = train(generator, discriminator, gen_optimiser, disc_optimiser, train_loader, batch_size, latent_size, f)
        fig = sample(generator)
        img_p = "mnist_cGAN_Conv_epoch" + str(f) + ".png"
        fig.savefig(img_p)
        loss_d_log = np.append(loss_d_log,loss_d)
        loss_g_log = np.append(loss_g_log,loss_g)
        plt.figure(f)
        plt.plot(loss_d_log,np.arange(0,len(loss_d_log),1))
        plt.axis([0,3000, 0, 1])
        plt.pause(0.001)
        plt.show()


if __name__ == "__main__":
    main()
