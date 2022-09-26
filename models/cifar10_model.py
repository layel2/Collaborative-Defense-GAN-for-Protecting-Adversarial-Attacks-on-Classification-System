import torch
from torch import nn
import torch.nn.functional as F


class cifar10_a(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2)
        self.dense1 = nn.Linear(in_features=256 * 2 * 2, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=256)
        self.dense3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.dropout(x, 0.25)
        x = nn.Flatten()(x)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, 0.5)
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dense3(x)

        return x


class cifar10_b(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)
        self.dense1 = nn.Linear(in_features=512 * 3 * 3, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.dropout(x, 0.25)
        x = nn.Flatten()(x)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, 0.5)
        x = self.dense2(x)

        return x

def getmodel(model=cifar10_a(), load_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not load_path is None:
        model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    return model