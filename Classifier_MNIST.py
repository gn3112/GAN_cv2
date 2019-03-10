import torchvision
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
from scipy.stats import entropy


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        # Pooling layers and batch norm
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(16,120,5)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = (F.softmax(self.fc2(x), dim=-1)).log()
        return x

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

def inception_score(classifier, data_path):
    batch_size = 256
    n_split = 10

    data = datasets.ImageFolder(root=data_path, transform=transform)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    proba_all = []
    label_all = []
    for batch_idx, (img, label) in enumerate(data_loader):
        proba = classifier(b)
        proba = proba.numpy()

        proba_all.append(proba)
        label_all.append(label)

    # inception score
    py = np.mean(proba_all, axis=0)
    n_img = proba_all.shape[0]
    n_img_split = n_img//n_split
    scores = []
    scores_splits = []
    for split in range(n_split):
        for i in range(split*n_img_split,(1+split)*n_img_split):
            pyx = proba[i,:]
            scores.append(entropy(pyx,py))

        scores_splits.append(np.exp(np.mean(scores)))

    # accuracy
    _, idx = proba_all.max(dim=1)
    n_correct = ((idx - label_all) == 0).sum(dim=0)
    accuracy = n_correct.numpy()/(n_img)

    return np.mean(scores_splits), np.std(scores_splits), accuracy

def export_nn_model(model,path):
    torch.save(model.state_dict(),path)
    model.load_state_dict(torch.load(path))
    model.eval()

def main():
    epoch = 10
    batch_size = 256

    data_path = os.path.join(os.path.expanduser('~'), '.torch', 'datasets', 'mnist')

    transform = transforms.Compose(
    [transforms.Resize(size=(32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

    train_data = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_path, train=False, transform=transform)
    indices = list(range(len(train_data)))
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, num_workers=4,sampler=SubsetRandomSampler(indices[:60000]))
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

    classifier = Classifier()
    optimiser = optim.Adam(classifier.parameters(), lr=2e-3, betas=(0.5, 0.999))

    for ep in range(epoch):
        train(classifier, optimiser, train_loader,ep)

    test(classifier, test_loader)
    inception_score(classifier, data_path="/fake_mnist/")


if __name__ == "__main__":
    main()
