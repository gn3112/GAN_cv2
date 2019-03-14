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
from CGAN import generator
import time

n_img = 1000
batch_size = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.isdir('CGAN_20ep_fake/'):
    os.mkdir('CGAN_20ep_fake/')

for label in range(10):
    if not os.path.isdir('CGAN_20ep_fake/' + str(label)):
        os.mkdir('CGAN_20ep_fake/' + str(label))

generator = generator(100)
generator.load_state_dict(torch.load('generator_param.pkl'))
generator.eval()

y_label = torch.tensor([])
for digits in range(10):
    sgl_label = torch.zeros(batch_size//10,10)
    sgl_label[:,digits] = 1
    # sgl_label = sgl_label.view(-1,10,1,1).float()
    y_label = torch.cat([y_label,sgl_label],0)


for epoch in range(n_img//batch_size):
    z_samples = torch.randn(batch_size, 100)
    start = time.time()
    images = generator(z_samples.to(device), y_label.to(device))
    images = torch.reshape(images, (batch_size,1,28,-1))
    end = time.time()
    print(end-start)
    label = 0
    for img in range((images.size())[0]-1):
        if img % (batch_size//10) == 0 and img!=0:
            label += 1
        path = 'CGAN_20ep_fake/' + str(label) + '/' + str(epoch) + '_' + str(img) + '.png'
        save_image(images[img,],path)

    print('Saved: ' + str((epoch+1)*batch_size) + ' images ')
