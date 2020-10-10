import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class VGG(BaseNet):
    def __init__(self, rep_dim=245):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False, padding=1)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 3, bias=False, padding=1)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(128, 256, 3, bias=False, padding=1)
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(256, 256, 3, bias=False, padding=1)
        self.bn2d5 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(256 * 7 * 7, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))  # 112
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))  # 56
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))  # 28
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))  # 14
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn2d5(x)))  # 7
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x


class VGG_Decoder(BaseNet):
    def __init__(self, rep_dim=245):
        super().__init__()

        self.rep_dim = rep_dim

        self.deconv0 = nn.ConvTranspose2d(int(self.rep_dim / (7 * 7)), 256, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv0.weight, gain=nn.init.calculate_gain("leaky_relu"))
        self.bn2d0 = nn.BatchNorm2d(256, eps=1e-04, affine=False)

        self.deconv0_1 = nn.ConvTranspose2d(256, 256, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv0_1.weight, gain=nn.init.calculate_gain("leaky_relu"))
        self.bn2d0_1 = nn.BatchNorm2d(256, eps=1e-04, affine=False)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain("leaky_relu"))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain("leaky_relu"))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain("leaky_relu"))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain("leaky_relu"))

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / (7 * 7)), 7, 7)
        x = F.leaky_relu(x)
        x = self.deconv0(x)
        x = F.interpolate(F.leaky_relu(self.bn2d0(x)), scale_factor=2)  # 14
        x = self.deconv0_1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d0_1(x)), scale_factor=2)  # 28
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)  # 56
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)  # 112
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)  # 224
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x


class VGG_Autoencoder(BaseNet):
    def __init__(self, rep_dim=245):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = VGG(rep_dim=rep_dim)
        self.decoder = VGG_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
