# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 25/08/2021 16:03
@Author: XINZHI YAO
"""

import os
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

# https://towardsdatascience.com/reparameterization-trick-126062cfd3c3


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_dif = '../checkpoint'
if not os.path.exists(checkpoint_dif):
    os.makedirs(checkpoint_dif)

# Hyper-parameters
# 28*28
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 15
batch_size = 128
learning_rate = 1e-3

# 1. load MNIST dataset
dataset = torchvision.datasets.MNIST(root='../../../data/minist',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# data.shape
# Out[23]: torch.Size([128, 1, 28, 28])
# label.shape
# Out[24]: torch.Size([128])
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

# Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, _image_size, _h_dim=400, _z_dim=20):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(_image_size, _h_dim)
        self.fc2 = nn.Linear(_h_dim, _z_dim) # Mean
        self.fc3 = nn.Linear(_h_dim, _z_dim) # Variance

        # Decoder
        self.fc4 = nn.Linear(_z_dim, _h_dim)
        self.fc5 = nn.Linear(_h_dim, _image_size)

    def encoder(self, _x):
        h = F.relu(self.fc1(_x))
        mean = self.fc2(h)
        # why we estimate log_var instead of var.
        # https://stats.stackexchange.com/questions/486158/reparameterization-trick-in-vaes-how-should-we-do-this/486161#486161
        log_variance = self.fc3(h)
        return mean, log_variance

    @staticmethod
    def reparameterize(mu, log_var):
        # exp(0.5*log(var)) = standard deviation
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        # grey scale
        # decoded = F.sigmoid(self.fc5(h))
        decoded = torch.sigmoid(self.fc5(h))
        return decoded

    @staticmethod
    def total_loss(_x_reconstruction, _x, mu, log_var):
        # https://stats.stackexchange.com/questions/485488/should-reconstruction-loss-be-computed-as-sum-or-average-over-input-for-variatio
        # fixme
        reconstruction_loss = F.binary_cross_entropy(_x_reconstruction, _x, reduction='sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = reconstruction_loss + kl_div
        # return reconstruction_loss
        # return kl_div
        return total_loss

    def forward(self, _x):
        _mu, _log_var = self.encoder(_x)
        _z = self.reparameterize(_mu, _log_var)
        _x_reconstruction = self.decode(_z)
        batch_loss = self.total_loss(_x_reconstruction, _x, _mu, _log_var)
        return _x_reconstruction, _mu, _log_var, batch_loss


# model train
model = VAE(image_size, h_dim, z_dim).to(device)

optimizer = Adam(model.parameters(), lr=learning_rate)
x = ''
for epoch in range(num_epochs):
    for step, (image, _) in enumerate(data_loader):
        optimizer.zero_grad()

        # 128, 784
        x = image.to(device).view(-1, image_size)

        _, _, _, loss = model(x)

        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f'Epoch: {epoch}, Step: {step}, loss: {loss:.4f}.')

    # save test image
    with torch.no_grad():
        # random image
        z = torch.randn(batch_size, z_dim).to(device)
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join(checkpoint_dif, f'random-{epoch}-kl+rc.png'))

        # reconstruction image
        x_reconstruction, _, _, _ = model(x)
        cat_image = torch.cat([x.view(-1, 1, 28, 28), x_reconstruction.view(-1, 1, 28, 28)], dim=3)
        save_image(cat_image, os.path.join(checkpoint_dif, f'reconst-{epoch}-kl+rc.png'))

