import torch
from torch import nn
import torch.nn.functional as F


class mnistmodel_A(nn.Module):
    def __init__(self):
        super(mnistmodel_A, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.dense1 = nn.Linear(in_features=64 * 12 * 12, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x,2)
        x = F.dropout(x, 0.25)
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, 0.5)
        x = self.dense2(x)

        return x



class mnistmodel_B(nn.Module):
    def __init__(self):
        super(mnistmodel_B, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1)
        self.dense1 = nn.Linear(in_features=128, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.dropout(x, 0.2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, 0.5)
        x = self.dense2(x)
        return x


class mnistmodel_C(nn.Module):
    def __init__(self):
        super(mnistmodel_C, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2)
        self.dense1 = nn.Linear(in_features=12 * 12 * 64, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 12 * 12 * 64)
        x = F.dropout(x, 0.25)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, 0.5)
        x = self.dense2(x)
        return x


class mnistmodel_B_old(nn.Module):
    def __init__(self):
        super(mnistmodel_B_old, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1)
        self.dense1 = nn.Linear(in_features=128, out_features=120)
        self.dense2 = nn.Linear(in_features=120, out_features=84)
        self.dense3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.dropout(x, 0.2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class mnistmodel_C_old(nn.Module):
    def __init__(self):
        super(mnistmodel_C_old, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2)
        self.dense1 = nn.Linear(in_features=12 * 12 * 64, out_features=120)
        self.dense2 = nn.Linear(in_features=120, out_features=84)
        self.dense3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 12 * 12 * 64)
        x = F.dropout(x, 0.25)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.dropout(x, 0.5)
        x = self.dense3(x)
        return x


def getmodel(model=mnistmodel_A(), load_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not load_path is None:
        model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    return model
