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

class Generator(nn.Module):
    def __init__(self,latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.conv1_z = nn.ConvTranspose2d((self.latent_size), 256, 4, stride=1)
        self.bn1_z = nn.BatchNorm2d(256)
        self.conv1_l = nn.ConvTranspose2d(10, 256, 4, stride=1)
        self.bn1_l = nn.BatchNorm2d(256)
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1)


    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, z, y_label):
        z = z.view(-1,self.latent_size,1,1)
        z = F.relu(self.bn1_z(self.conv1_z(z)))
        y = F.relu(self.bn1_l(self.conv1_l(y_label)))
        x = torch.cat([z,y],1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return torch.tanh(self.conv4(x))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_z = nn.Conv2d(1, 64, 4, stride=2, padding=1)
        self.conv1_l = nn.Conv2d(10, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1, 4, stride=1, padding=0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, z, y_fill):
        z = F.leaky_relu(self.conv1_z(z), 0.2)
        y = F.leaky_relu(self.conv1_l(y_fill), 0.2)
        x = torch.cat([z,y],1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        return torch.sigmoid(self.conv4(x)).view(-1)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def create_directories():
    dir_path = os.getcwd()
    if not os.path.exists("fake_mnist"):
        os.makedirs("fake_mnist")

    for digits in range(10):
        if not os.path.exists("fake_mnist/"+str(digits)):
            os.makedirs(dir_path + "/fake_mnist/" + str(digits))

def new_dataset(n_img,digit):
    dir_path = os.getcwd()

    for i in range(n_img):
        save_image(n_img[i], dir_path +  "/fake_mnist/" + str(digit) + "img" + str(i) +".png")

def export_nn_model(model,path):
    torch.save(model.state_dict(),path)
    model.load_state_dict(torch.load(path))
    model.eval()

def out_size_tconv(in_size,stride,padding,kernel,output_padding):
    return (in_size - 1) * stride - 2 * padding + kernel + output_padding

def train(generator, discriminator, gen_optimiser, disc_optimiser, train_loader, batch_size, latent_size, ep, device):
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
        real_loss = F.binary_cross_entropy(real_y, torch.ones_like(real_y))
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
        gen_optimiser.zero_grad()

        # Train generator to fool discriminator on fake data
        fake_y = discriminator(fake_x.to(device),y_fill.to(device))
        fake_loss = F.binary_cross_entropy(fake_y, torch.ones_like(fake_y))
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
    return loss_d, loss_g,train_time

def sample(generator,device):
    n_img = 100
    generator.eval()
    black_bar = torch.zeros(3, 10 * 32, 20)
    with torch.no_grad():
        z_samples = torch.randn(n_img, 100)
        z_interp = torch.zeros(n_img, 100)
        y_label = torch.tensor([])
        for digits in range(10):
            sgl_label = torch.zeros(n_img//10,10)
            sgl_label[:,digits] = 1
            sgl_label = sgl_label.view(-1,10,1,1).float()
            y_label = torch.cat([y_label,sgl_label],0)

        samples = make_grid(generator(z_samples.to(device),y_label.to(device)),nrow=10, padding=2)
        # interps = make_grid(generator(z_interp.to(device),y_label.to(device)),nrow=6, padding=0)
        samples = ((samples.cpu())+1)/2
        # interps = ((interps.cpu())+1)/2

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_axis_off()
        ax.imshow(np.transpose((samples).numpy(), [1, 2, 0]))
    return fig

def export_nn_model(model,path):
    torch.save(model.state_dict(),path)
    # model.load_state_dict(torch.load(path))
    # model.eval()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else print("cuda not available"))

    create_directories()
    # f = open("cDCGAN_log.txt","w+")

    plt.interactive(True)
    latent_size = 100
    batch_size = 128
    epoch = 20
    transform = transforms.Compose(
    [transforms.Resize(size=(32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
     ])

    data_path = os.path.join(os.path.expanduser('~'), '.torch', 'datasets', 'mnist')
    train_data = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_path, train=False, transform=transform)

    indices = list(range(len(train_data)))
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, num_workers=4,sampler=SubsetRandomSampler(indices[:60000]))
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

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
        loss_d, loss_g, train_time = train(generator, discriminator, gen_optimiser, disc_optimiser, train_loader, batch_size, latent_size, f, device)
        loss_d_log = np.append(loss_d_log,loss_d)
        loss_g_log = np.append(loss_g_log,loss_g)

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
        fig.savefig('cDCGAN/epoch'+str(f+1)+'.png')

        # log model info
        # f.write(train_time)
    plt.figure()
    plt.plot(np.arange(0,len(loss_d_log),1),loss_d_log, label='D_loss')
    plt.plot(np.arange(0,len(loss_g_log),1),loss_g_log, label='G_loss')
    plt.legend(loc=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cDCGAN/loss.png')

    export_nn_model(generator.cpu(),'model_cDCGAN.pth')


if __name__ == "__main__":
    main()
