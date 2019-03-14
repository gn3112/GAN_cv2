import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from matplotlib.ticker import MaxNLocator
from torchvision.utils import make_grid, save_image
import numpy as np

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, latent_size):
        super(generator, self).__init__()
        self.latent_size = latent_size
        self.fc1_1 = nn.Linear(latent_size, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(10, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 784)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.tanh(self.fc4(x))
        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1_1 = nn.Linear(784, 1024)
        self.fc1_2 = nn.Linear(10, 1024)
        self.fc2 = nn.Linear(2048, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.fc1_1(input), 0.2)
        y = F.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = F.sigmoid(self.fc4(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def save_results(p, G, fixed_z_, fixed_y_label_, show = False, save = False, path = 'results.png',device='s'):

    G.eval()
    test_images = G(fixed_z_.to(device), fixed_y_label_.to(device))
    G.train()

    for j in range(len(test_images)):
        plt.imshow(test_images[j].cpu().data.view(28, 28).numpy(), cmap='gray')
        if j < 10:
            real_path = path + '0/' + str(j) + '_' +str(p) + '.png'
        elif j > 9 and j < 20:
            real_path = path + '1/' + str(j) + '_' +str(p) + '.png'
        elif j > 19 and j < 30:
            real_path = path + '2/' + str(j) + '_' +str(p) + '.png'
        elif j > 29 and j < 40:
            real_path = path + '3/' + str(j) + '_' +str(p) + '.png'
        elif j > 39 and j < 50:
            real_path = path + '4/' + str(j) + '_' +str(p) + '.png'
        elif j > 49 and j < 60:
            real_path = path + '5/' + str(j) + '_' +str(p) + '.png'
        elif j > 59 and j < 70:
            real_path = path + '6/' + str(j) + '_' +str(p) + '.png'
        elif j > 69 and j < 80:
            real_path = path + '7/' + str(j) + '_' +str(p) + '.png'
        elif j > 79 and j < 90:
            real_path = path + '8/' + str(j) + '_' +str(p) + '.png'
        elif j > 89:
            real_path = path + '9/' + str(j) + '_' +str(p) + '.png'

        plt.savefig(real_path)



def show_result(num_epoch, G, fixed_z_, fixed_y_label_, show = False, save = False, path = 'result.png', device= 's'):

    G.eval()
    test_images = G(fixed_z_.to(device), fixed_y_label_.to(device))
    G.train()

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k].cpu().data.view(28, 28).numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        fig = sample(generator,device)
        fig.savefig('cDCGAN/epoch'+str(f+1)+'.png')

    if show:
        plt.show()
    else:
        plt.close()

def train(G, D, G_optimizer, D_optimizer, train_loader, epoch, BCE_loss, device):
    D_losses = []
    G_losses = []
    for batch_idx, (x_, y_) in enumerate(train_loader):
        # train discriminator D
        D.zero_grad()

        mini_batch = x_.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)
        y_label_ = torch.zeros(mini_batch, 10)
        y_label_.scatter_(1, y_.view(mini_batch, 1), 1)

        x_ = x_.view(-1, 28 * 28)
        x_, y_label_, y_real_, y_fake_ = Variable(x_), Variable(y_label_), Variable(y_real_), Variable(y_fake_)
        D_result = D(x_.to(device), y_label_.to(device)).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_.to(device))

        z_ = torch.rand((mini_batch, 100))
        y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor)
        y_label_ = torch.zeros(mini_batch, 10)
        y_label_.scatter_(1, y_.view(mini_batch, 1), 1)

        z_, y_label_ = Variable(z_), Variable(y_label_)


        G_result = G(z_.to(device), y_label_.to(device))

        D_result = D(G_result.to(device), y_label_.to(device)).squeeze()
        D_fake_loss = BCE_loss(D_result, y_fake_.to(device))
        D_fake_score = D_result.data.mean()

        D_loss = D_real_loss + D_fake_loss

        D_loss.backward()
        D_optimizer.step()

        # train generator G
        G.zero_grad()

        z_ = torch.rand((mini_batch, 100))
        y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor)
        y_label_ = torch.zeros(mini_batch, 10)
        y_label_.scatter_(1, y_.view(mini_batch, 1), 1)

        z_, y_label_ = Variable(z_), Variable(y_label_)

        G_result = G(z_.to(device), y_label_.to(device))
        D_result = D(G_result.to(device), y_label_.to(device)).squeeze()
        G_loss = BCE_loss(D_result, y_real_.to(device))
        G_loss.backward()
        G_optimizer.step()

        D_losses.append(D_loss)
        G_losses.append(G_loss)

    loss_d = 1/(batch_idx+1) * sum(D_losses)
    loss_g = 1/(batch_idx+1) * sum(G_losses)
    return loss_d, loss_g

def sample(generator, device):
    n_img = 100
    generator.eval()
    z_samples = torch.randn(n_img, 100)
    with torch.no_grad():
        z_interp = torch.zeros(n_img, 10)
        y_label = torch.tensor([])
        for digits in range(10):
            sgl_label = torch.zeros(n_img//10,10)
            sgl_label[:,digits] = 1
            y_label = torch.cat([y_label,sgl_label],0)

        img = generator(z_samples.to(device),y_label.to(device)) #.unsqueeze(-1)
        img = torch.reshape(img, (100,1,28,-1))
        samples = make_grid(img,nrow=10, padding=2)
        # interps = make_grid(generator(z_interp.to(device),y_label.to(device)),nrow=6, padding=0)
        samples = ((samples.cpu())+1)/2
        # interps = ((interps.cpu())+1)/2
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_axis_off()
        print((np.transpose((samples).numpy(), [1, 2, 0])).shape)
        ax.imshow(np.transpose((samples).numpy(), [1, 2, 0]))
    return fig
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    latent_size = 100
    batch_size = 128
    epochs = 50
    # data_loader
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
     ])

    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

    temp_z_ = torch.rand(10, 100)
    fixed_z_ = temp_z_
    fixed_y_ = torch.zeros(10, 1)
    for i in range(9):
        fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
        temp = torch.ones(10,1) + i
        fixed_y_ = torch.cat([fixed_y_, temp], 0)


    fixed_z_ = Variable(fixed_z_,requires_grad=False)
    fixed_y_label_ = torch.zeros(100, 10)
    fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
    fixed_y_label_ = Variable(fixed_y_label_, requires_grad=False)

    # network
    G = generator(latent_size)
    D = discriminator()
    G.to(device)
    D.to(device)
    G.weight_init(mean=0, std=0.02)
    D.weight_init(mean=0, std=0.02)

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()
    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr= 0.0002, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr= 0.0002, betas=(0.5, 0.999))

    # results save folder
    if not os.path.isdir('MNIST_cGAN_results'):
        os.mkdir('MNIST_cGAN_results')
    if not os.path.isdir('MNIST_cGAN_results/Fixed_results'):
        os.mkdir('MNIST_cGAN_results/Fixed_results')
    for k in range(10):
        if not os.path.isdir('MNIST_cGAN_results/' + str(k)):
            os.mkdir('MNIST_cGAN_results/' + str(k))

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    print('training start!')
    start_time = time.time()
    D_losses = []
    G_losses = []
    for epoch in range(epochs):
        print("Epoch: " + str(epoch+1))
        epoch_start_time = time.time()
        # learning rate decay
        if (epoch+1) == 30:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        if (epoch+1) == 40:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        D_loss, G_loss = train(G, D, G_optimizer, D_optimizer, train_loader, epoch, BCE_loss, device)

        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        fig2 = plt.figure()
        ax2 = fig2.gca()
        plt.plot(D_losses, label='D_loss')
        plt.plot(G_losses, label='G_loss')
        plt.legend(loc=4)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig('MNIST_cGAN_results/loss.png')
        plt.close(fig2)

        fig = sample(G,device)
        fig.savefig('MNIST_cGAN_results/epoch'+str(epoch+1)+'.png')
        plt.close(fig)

        if epoch == 19:
            torch.save(G.state_dict(), "MNIST_cGAN_results/generator_param_ep20.pth")


    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)


    # Storing labeled, generated images in folder:
    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), epochs, total_ptime))
    print("Training finish!... save training results")
    torch.save(G.state_dict(), "MNIST_cGAN_results/generator_param.pth")


if __name__ == "__main__":
    main()
