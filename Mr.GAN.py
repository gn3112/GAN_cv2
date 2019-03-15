import math
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import time
from matplotlib.ticker import MaxNLocator
from CGAN_conv import Generator, Discriminator,export_nn_model, normal_init, sample
from Classifier_MNIST import Classifier


def train(generator, discriminator, classifier, gen_optimiser, disc_optimiser, train_loader, batch_size, latent_size, ep, device):
    start_time = time.time()
    loss_g = []
    loss_d = []
    for batch_idx, (data_img, label) in enumerate(train_loader):
        real_x = data_img.to(device)
        y_label = torch.zeros(batch_size,10)
        y_label[range(batch_size),label] = 1
        y_label = y_label.view(-1,10,1,1).float()

        y_fill = None
        y_fill = y_label * torch.ones(batch_size,10,32,32)
        y_fill = y_fill.view(-1,10,32,32).float()
        disc_optimiser.zero_grad()
        # Train discriminator to identify real data
        real_y = discriminator(real_x,y_fill.to(device))
        real_loss = F.binary_cross_entropy(real_y, torch.ones_like(real_y)*0.9)
        real_loss.backward()

        # Train discriminator to identify fake data
        data_z = torch.randn(batch_size, latent_size)
        noise = data_z.to(device)
        fake_x = generator(noise,y_label.to(device))
        fake_y = discriminator(fake_x.detach().to(device),y_fill.to(device))
        fake_loss = F.binary_cross_entropy(fake_y, torch.zeros_like(fake_y))
        fake_loss.backward()
        loss = (fake_loss + real_loss).detach().cpu().numpy()
        loss_d = np.append(loss_d, loss)
        disc_optimiser.step()

        # Train generator to fool discriminator on fake data
        gen_optimiser.zero_grad()
        fake_y = discriminator(fake_x.to(device),y_fill.to(device))
        logits = classifier(fake_x)
        aux_loss = F.cross_entropy(logits,label)
        aux_loss.backward(retain_graph=True)
        fake_loss = F.binary_cross_entropy(fake_y, torch.ones_like(fake_y)*0.9)
        fake_loss.backward()
        gen_optimiser.step()
        loss = fake_loss.detach().cpu().numpy()
        loss_g = np.append(loss_g, loss)

        if batch_idx % 50 == 0 and batch_idx !=0:
            print("---- Iteration: " + str(batch_idx) + " in Epoch: " + str(ep+1) + " ----")
            print("---- Loss Generator: " + str(np.around(loss_g[-1],decimals=2)) + " Loss Discriminator: " + str(np.around(loss_d[-1],decimals=2)) + "----")

    end_time = time.time()
    train_time = end_time - start_time
    print("Training time: ", train_time)
    loss_d = 1/(batch_idx+1) * sum(loss_d)
    loss_g = 1/(batch_idx+1) * sum(loss_g)
    return loss_d, loss_g,train_time


def main():
    classifier = Classifier()
    classifier.load_state_dict(torch.load("classifier_mnist.pth"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    latent_size = 100
    batch_size = 128
    epoch = 2
    transform = transforms.Compose(
    [transforms.Resize(size=(32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
     ])

    data_path = os.path.join(os.path.expanduser('~'), '.torch', 'datasets', 'mnist')
    train_data = datasets.MNIST(data_path, train=True, download=True, transform=transform)

    indices = list(range(len(train_data)))
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, num_workers=4,sampler=SubsetRandomSampler(indices[:1000]))

    generator = Generator(latent_size)
    generator.to(device)
    discriminator = Discriminator()
    discriminator.to(device)
    generator.weight_init(mean=0.0, std=0.02)
    discriminator.weight_init(mean=0.0, std=0.02)

    gen_optimiser = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    disc_optimiser = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loss_d_log = []
    loss_g_log = []
    for f in range(epoch):
        loss_d, loss_g, train_time = train(generator, discriminator, classifier, gen_optimiser, disc_optimiser, train_loader, batch_size, latent_size, f, device)
        loss_d_log.append(loss_d)
        loss_g_log.append(loss_g)

            # learning rate decay
        if (f+1) == 11:
            gen_optimiser.param_groups[0]['lr'] /= 10
            disc_optimiser.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        if (f+1) == 16:
            gen_optimiser.param_groups[0]['lr'] /= 10
            disc_optimiser.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        fig = sample(generator,device)
        fig.savefig('Mr_GAN/epoch'+str(f+1)+'.png')

        # log model info
        # f.write(train_time)

        fig2 = plt.figure()
        ax2 = fig2.gca()
        plt.plot(loss_d_log, label='D_loss')
        plt.plot(loss_g_log, label='G_loss')
        plt.legend(loc=4)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig('Mr_GAN/loss.png')
        plt.close(fig2)

    export_nn_model(generator.cpu(),'Mr_GAN.pth')


if __name__ == "__main__":
    main()
