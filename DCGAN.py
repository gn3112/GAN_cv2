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
import time
from matplotlib.ticker import MaxNLocator

class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.d = 1
        self.latent_size = latent_size
        self.conv1 = nn.ConvTranspose2d(latent_size, 32*self.d, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(32*self.d)
        self.conv2 = nn.ConvTranspose2d(32*self.d, 16*self.d, 4, stride=2, padding=2, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16*self.d)
        self.conv3 = nn.ConvTranspose2d(16*self.d, 8*self.d, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(8*self.d)
        self.conv4 = nn.ConvTranspose2d(8*self.d, 1, 4, stride=2, padding=1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, z):
        z = z.view(-1, self.latent_size, 1, 1)
        x = F.relu(self.bn1(self.conv1(z)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return torch.tanh(self.conv4(x))

        # ref_x = ref_z.view(-1, self.latent_size, 1, 1)
        # ref_x = self.conv1(ref_x)
        # ref_x, mean1, meansq1 = self.vbn1(ref_x, None, None)
        # ref_x = F.relu(ref_x)
        # ref_x = self.conv2(ref_x)
        # ref_x, mean2, meansq2 = self.vbn2(ref_x, None, None)
        # ref_x = F.relu(ref_x)
        # ref_x = self.conv3(ref_x)
        # ref_x, mean3, meansq3 = self.vbn3(ref_x, None, None)
        #
        # x = z.view(-1, self.latent_size, 1, 1)
        # x = self.conv1(x)
        # x, _, _ = self.vbn1(x, mean1, meansq1)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x, _, _ = self.vbn2(x, mean2, meansq2)
        # x = F.relu(x)
        # x = self.conv3(x)
        # x, _, _ = self.vbn3(x, mean3, meansq3)
        # x = F.relu(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Takes input of 32x32x1
        # Skipping 2 pixels
        self.d = 1
        self.conv1 = nn.Conv2d(1, 8*self.d, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8*self.d, 16*self.d, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16*self.d)
        self.conv3 = nn.Conv2d(16*self.d, 32*self.d, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32*self.d)
        self.conv4 = nn.Conv2d(32*self.d, 64*self.d, 4, stride=2, padding=1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        return torch.sigmoid(self.conv4(x)).view(-1)



def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)

def train(generator, discriminator, gen_optimiser, disc_optimiser, train_loader, batch_size, latent_size, ep):
    loss_g = []
    loss_d = []
    start_time = time.time()
    for batch_idx, (real_x, _) in enumerate(train_loader):
        if (ep == 0 or ep == 5) and batch_idx<8:
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
            loss = (fake_loss + real_loss).detach()
            loss_d.append(loss)
            disc_optimiser.step()
        else:
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
            loss = (fake_loss + real_loss).detach()
            loss_d.append(loss)
            disc_optimiser.step()
            gen_optimiser.zero_grad()
            # Train generator to fool discriminator on fake data
            fake_y = discriminator(fake_x)
            fake_loss = F.binary_cross_entropy(fake_y, torch.ones_like(fake_y))
            fake_loss.backward()
            gen_optimiser.step()
            loss = fake_loss.detach()
            loss_g.append(loss)

        if batch_idx % 50 == 0 and batch_idx !=0:
            print("---- Iteration: " + str(batch_idx) + " in Epoch: " + str(ep+1) + " ----")
            print("---- Loss Generator: " + str(np.around(loss_g[-1],decimals=2)) + " Loss Discriminator: " + str(np.around(loss_d[-1],decimals=2)) + "----")

    end_time = time.time()
    train_time = end_time - start_time
    print("Training time in epoch: ", train_time)

    loss_d = 1/(batch_idx+1) * sum(loss_d)
    loss_g = 1/(batch_idx+1) * sum(loss_g)
    return loss_d, loss_g

def sample(generator):
    n_img = 100
    generator.eval()
    black_bar = torch.zeros(3, 10 * 32, 20)
    with torch.no_grad():
        z_samples = torch.randn(n_img, 10)
        z_interp = torch.zeros(n_img, 10)

        samples = make_grid(generator(z_samples),nrow=10, padding=2)
        # interps = make_grid(generator(z_interp.to(device),y_label.to(device)),nrow=6, padding=0)
        samples = ((samples+1)/2)
        # interps = ((interps.cpu())+1)/2

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_axis_off()
        ax.imshow(np.transpose((samples).numpy(), [1, 2, 0]))
    return fig

def main():
    if not os.path.exists("DCGAN"):
        os.makedirs("DCGAN")

    latent_size = 10
    batch_size = 64
    epoch = 20

    transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
     ])

    data_path = os.path.join(os.path.expanduser('~'), '.torch', 'datasets', 'mnist')
    train_data = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_path, train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

    generator = Generator(latent_size)
    discriminator = Discriminator()
    gen_optimiser = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.9))
    disc_optimiser = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))
    generator.weight_init(mean=0.0, std=0.02)
    discriminator.weight_init(mean=0.0, std=0.02)

    plt.axis('off')
    loss_d_log = []
    loss_g_log = []
    for f in range(epoch):
        loss_d, loss_g = train(generator, discriminator, gen_optimiser, disc_optimiser, train_loader, batch_size, latent_size, f)
        fig = sample(generator)
        img_p = "DCGAN/mnist_epoch" + str(f+1) + ".png"
        fig.savefig(img_p)
        plt.close(fig)

        loss_d_log = np.append(loss_d_log,loss_d)
        loss_g_log = np.append(loss_g_log,loss_g)
        fig2 = plt.figure()
        ax2 = fig2.gca()
        plt.plot(loss_d_log, label='D_loss')
        plt.plot(loss_g_log, label='G_loss')
        plt.legend(loc=4)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig('DCGAN/loss.png')
        plt.close(fig2)
    # clear_output(wait=True)

if __name__ == "__main__":
    main()
