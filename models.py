import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[10, 5], num_channels=[15, 15], interm_dim=15):
        super(LossNet, self).__init__()

        self.GAP1 = nn.AvgPool1d(feature_sizes[0])
        self.GAP2 = nn.AvgPool1d(feature_sizes[1])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)

        self.linear = nn.Linear(2 * interm_dim, 1)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out = self.linear(torch.cat((out1, out2), 1))
        return out


class MolecularVAE(nn.Module):
    def __init__(self):
        super(MolecularVAE, self).__init__()

        self.conv_1 = nn.Conv1d(200, 18, kernel_size=18)
        self.conv_2 = nn.Conv1d(18, 18, kernel_size=18)
        self.conv_3 = nn.Conv1d(18, 20, kernel_size=22)
        self.linear_0 = nn.Linear(400, 435)
        self.linear_1 = nn.Linear(435, LATENT_DIM)
        self.linear_2 = nn.Linear(435, LATENT_DIM)

        self.linear_3 = nn.Linear(LATENT_DIM+1, 300)
        self.gru = nn.GRU(300, 500, 3, batch_first=True)
        self.linear_4 = nn.Linear(500, 75)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 200, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_4(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x, r):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        z = torch.cat([z, r], 1)
        return self.decode(z), z_mean, z_logvar


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

        self.fc1 = nn.Linear(LATENT_DIM, 150)
        self.fc2 = nn.Linear(150, 75)
        self.fc3 = nn.Linear(75, 2)

        self.relu = nn.ReLU()

    def forward(self, z_mean):
        out1 = self.relu(self.fc1(z_mean))
        out2 = self.relu(self.fc2(out1))
        out = self.relu(self.fc3(out2))

        return out, [out1, out2]


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""

    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim+1, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, r, z):
        z = torch.cat([z, r], 1)
        return self.net(z)
    # def forward(self, z):
    #     return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
