import torch
import torch.nn.functional as F
from functools import reduce


class ConvNet(torch.nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.C, self.H, self.W = din
        self.Dout = dout
        self.Chid = 32
        self.Chid2 = 64
        self.Chid3 = 64

        self.conv1 = torch.nn.Conv2d(
            in_channels=self.C,
            out_channels=self.Chid,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=self.Chid,
            out_channels=self.Chid2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=self.Chid2,
            out_channels=self.Chid3,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = torch.nn.Linear(int(self.Chid3 * self.H * self.W / 16), 564)
        self.fc2 = torch.nn.Linear(564, dout)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(batch_size, int(self.Chid3 * self.H * self.W / 16))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.din = reduce(lambda x, y: x * y, din)
        self.l1 = torch.nn.Linear(self.din, 1024)
        self.l2 = torch.nn.Linear(1024, 1024)
        self.l3 = torch.nn.Linear(1024, 1024)
        # self.l4 = torch.nn.Linear(512, 1024)
        self.lout = torch.nn.Linear(1024, dout)

    def forward(self, x):
        x = x.view(-1, self.din)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        # x = F.relu(self.l4(x))
        return self.lout(x)
