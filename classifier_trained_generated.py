import torchvision
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import os
from scipy.stats import entropy
import numpy as np
import random
from Classifier_MNIST import Classifier
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt

classifier = Classifier()

def test(classifier, test_loader):
    n_correct = 0
    for batch_idx, (img, label) in enumerate(test_loader):
        proba = classifier(img)
        proba = proba.exp()
        _, idx = proba.max(dim=1)
        n_correct += ((idx - label) == 0).sum(dim=0)

    accuracy = n_correct.numpy()/(10000)
    print("Accuracy on 10K test: ",accuracy)
    return accuracy

def train(classifier, optimiser, train_loader,ep):
    for batch_idx, (img, label) in enumerate(train_loader):
        optimiser.zero_grad()
        proba = classifier(img)
        loss = F.cross_entropy(proba, label)
        loss.backward()
        optimiser.step()
        if batch_idx % 100 == 0 and batch_idx !=0:
            print("---- Iteration: " + str(batch_idx) + " in Epoch: " + str(ep+1) + " Loss: " + str(loss.detach().numpy()) + " ----")

def main():
    # epoch = 10
    # batch_size = 128
    # fine_tune = True
    #
    # data_path = os.path.join(os.path.expanduser('~'), '.torch', 'datasets', 'mnist')
    #
    #
    # transform = transforms.Compose(
    # [transforms.Resize(size=(32, 32)),
    # transforms.Grayscale(num_output_channels=1),
    # transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #  ])
    #  # 0% | 10% | 20% | 50% | 100% real
    # prop_real = [0, 6000, 12000, 30000, 60000]
    # prop_fake = [59000, 54000, 48000, 30000, 0]
    # accuracy_all = []
    #
    # test_data = datasets.MNIST(data_path, train=False, transform=transform)
    # test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)
    # train_data_real = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    # indices = list(range(len(train_data_real)))
    # random.shuffle(indices)
    #
    # fake_data = datasets.ImageFolder(root='cDCGAN_fake',
    #                                        transform=transform)
    # indices_f = list(range(len(fake_data)))
    # random.shuffle(indices_f)
    #
    # real_data = datasets.ImageFolder(root='real_image_mnist',
    #                                        transform=transform)
    # indices_r = list(range(len(real_data)))
    # random.shuffle(indices_r)
    #
    # accuracy_all_fine = []
    # accuracy_all_shuffle = []
    # for fine_tune in range(2):
    #
    #     if fine_tune == 0:
    #         for run in range(len(prop_real)):
    #             classifier = Classifier()
    #             optimiser = optim.Adam(classifier.parameters(), lr=2e-3, betas=(0.5, 0.999))
    #
    #             # train on fake images them
    #
    #             train_loader_real = DataLoader(train_data_real, batch_size=batch_size, drop_last=True, num_workers=4,sampler=SubsetRandomSampler(indices[:prop_real[run]]))
    #             train_loader_fake = DataLoader(fake_data, batch_size=batch_size, drop_last=True, num_workers=4,sampler=SubsetRandomSampler(indices_f[:prop_fake[run]]))
    #
    #             for ep in range(epoch):
    #                 if run == 0:
    #                     train(classifier, optimiser, train_loader_fake,ep)
    #                     print(run)
    #                 elif run == 4:
    #                     train(classifier, optimiser, train_loader_real,ep)
    #                     print(run)
    #                 elif ep < 7:
    #                     train(classifier, optimiser, train_loader_fake,ep)
    #                     print(run)
    #                 else:
    #                     train(classifier, optimiser, train_loader_real,ep)
    #                     print(run)
    #
    #
    #             accuracy = test(classifier, test_loader)
    #             accuracy_all_fine.append(accuracy)
    #
    #     elif fine_tune == 1:
    #         for run in range(len(prop_real)):
    #             classifier = Classifier()
    #             optimiser = optim.Adam(classifier.parameters(), lr=2e-3, betas=(0.5, 0.999))
    #
    #             data_real = Subset(real_data,indices_r[:prop_real[run]])
    #             data_fake = Subset(fake_data,indices_f[:prop_fake[run]])
    #             data_all = ConcatDataset((data_real,data_fake))
    #             train_loader = DataLoader(data_all, batch_size=batch_size, drop_last=True, num_workers=4, shuffle=True)
    #
    #             for ep in range(epoch):
    #                 train(classifier, optimiser, train_loader, ep)
    #
    #             accuracy = test(classifier, test_loader)
    #             accuracy_all_shuffle.append(accuracy)

    fig = plt.figure()
    ax2 = fig.gca()
    print(np.reshape(np.array([0, 10, 20, 50, 100]),(5,1)))
    plt.plot([0, 10, 20, 50, 100], [0.887, 0.92, 0.96,0.975,0.9904], label='Fine-tuning')
    plt.plot([0, 10, 20, 50, 100], [0.878,0.9975,0.9769,0.9851,0.9911], label='Shuffled')
    plt.legend(loc=4)
    plt.xlabel('Percentage of real data')
    plt.ylabel('Accuracy')
    plt.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('mixed_classifier.png')
    plt.close(fig)



if __name__ == "__main__":
    main()
