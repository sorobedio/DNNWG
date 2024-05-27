
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_classes):
        super(MLP, self).__init__()
        self.ncs = n_classes

        self.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                        nn.LeakyReLU(),
                                        nn.Linear(in_features=256, out_features=n_classes))

    def forward(self, x):
        out = self.classifier(x)
        return out

class Classifier(nn.Module):
    def __init__(self,in_dim=512, n_classes=10, bias=True):
        super(Classifier, self).__init__()
        self.ncs = n_classes
        self.in_dim=in_dim
        self.bias=bias

        self.classifier = nn.Sequential(nn.Linear(in_features=in_dim, out_features=n_classes, bias=bias))

    def forward(self, x):
        out = self.classifier(x)
        return out

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, n_classes=5):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        return self.linear(x)


class TinyCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_claases = n_classes
        self.conv= nn.Conv2d(2, 32, 3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self.l1 = nn.Linear(256, n_classes)
    def forward(self, x):
        x = x.reshape(-1, 2, 16, 16)
        x = self.conv(x)
        x = F.leaky_relu(x)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = x.view(-1, 256)
        out = self.l1(x)
        return out
