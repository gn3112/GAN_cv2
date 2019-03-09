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
        self.conv1 = nn.Conv2d(1,20,5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(20,50,5, stride=1, padding=0)
        self.fc1 = nn.Linear(4*4*50,500)
        self.fc2 = nn.Linear(500,100)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2).view(-1,4*4*50)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x



def test(classifier, test_loader):
    n_correct = 0
    for batch_idx, (img, label) in enumerate(test_loader):
        proba = classifier(img)
        _, idx = proba.max(dim=1)
        n_correct += ((idx - label) != 0).sum(dim=0)

    accuracy = n_correct.numpy()/((batch_idx+1)*64)
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
    epoch = 4
    batch_size = 128

    data_path = os.path.join(os.path.expanduser('~'), '.torch', 'datasets', 'mnist')

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_path, train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

    classifier = Classifier()
    optimiser = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.5, 0.999))

    for ep in range(epoch):
        train(classifier, optimiser, train_loader,ep)

    test(classifier, test_loader)


if __name__ == "__main__":
    main()
