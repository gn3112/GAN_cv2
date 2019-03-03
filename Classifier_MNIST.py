import torchvision
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Pooling layers and batch norm
        self.conv1 = nn.Conv2d(1,4,3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4,8,3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(8,5,3, stride=2, padding=1)
        self.fc = nn.Linear(7*7*5,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)).view(-1, 5 * 7 * 7)
        x = F.softmax(self.fc(x))

        return x



def test(classifier, test_loader):
    n_correct = 0
    for batch_idx, (img, label) in enumerate(test_loader):
        proba = classifier(img)
        _, idx = proba.max(dim=1)
        n_correct += ((idx - label) != 0).sum(dim=0)

    accuracy = n_correct.numpy()/((batch_idx+1)*64)
    return accuracy

def train(classifier, optimiser, train_loader):
    for batch_idx, (img, label) in enumerate(train_loader):
        proba = classifier(img)
        loss = F.cross_entropy(proba, label)
        loss.backward()
        optimiser.step()



def main():
    epoch = 1

    data_path = os.path.join(os.path.expanduser('~'), '.torch', 'datasets', 'mnist')

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_path, train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=64, num_workers=4)

    classifier = Classifier()
    optimiser = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.5, 0.999))

    for _ in range(epoch):
        train(classifier, optimiser, train_loader)

    test(classifier, test_loader)


if __name__ == "__main__":
    main()
