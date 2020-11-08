import torch
import torch.nn as nn
from torch.nn.functional import one_hot


class VAEMNIST(nn.Module):
    """Implementation of VAE model on mnist dataset"""
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
                            # (1, 28, 28)
                            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2),
                            # (32, 14, 14)
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2))
                            # (64, 7, 7)

        self.decoder = nn.Sequential(
                            # (64, 7, 7)
                            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                            # (32, 15, 15)
                            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2),
                            nn.Sigmoid())
                            # (1, 30, 30)

        self.mu = nn.Linear(64*7*7, latent_dim)
        self.log_var = nn.Linear(64*7*7, latent_dim)
        self.upsample = nn.Linear(latent_dim, 64*7*7)

    def forward(self, imgs):
        x = self.encoder(imgs)
        x = x.view(x.size(0), -1)

        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self._sampling(mu, log_var)

        x = self.upsample(z)
        x = x.view(x.size(0), 64, 7, 7)
        x = self.decoder(x)

        return x[:, :, 1:29, 1:29], mu, log_var

    def get_latent(self, imgs):
        x = self.encoder(imgs)
        x = x.view(x.size(0), -1)

        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self._sampling(mu, log_var)
        return z

    def generate(self, z):
        x = self.upsample(z)
        x = x.view(x.size(0), 64, 7, 7)
        x = self.decoder(x)
        return x[:, :, 1:29, 1:29]

    def _sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


class CVAEMNIST(nn.Module):
    """Implementation of conditional VAE on mnist dataset"""
    N_CLASSES = 10
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
                            # (1, 28, 28)
                            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2),
                            # (32, 14, 14)
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2))
                            # (64, 7, 7)

        self.decoder = nn.Sequential(
                            # (64, 7, 7)
                            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                            # (32, 15, 15)
                            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2),
                            nn.Sigmoid())
                            # (1, 30, 30)

        self.mu = nn.Linear(64*7*7+CVAEMNIST.N_CLASSES, latent_dim)
        self.log_var = nn.Linear(64*7*7+CVAEMNIST.N_CLASSES, latent_dim)
        self.upsample = nn.Linear(latent_dim+CVAEMNIST.N_CLASSES, 64*7*7)

    def forward(self, imgs, labels):
        x = self.encoder(imgs)
        x = x.view(x.size(0), -1)

        one_hots = one_hot(labels, CVAEMNIST.N_CLASSES).float()
        x = torch.cat([x, one_hots], axis=1)

        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self._sampling(mu, log_var)
        x = torch.cat([z, one_hots], axis=1)
        x = self.upsample(x)
        x = x.view(x.size(0), 64, 7, 7)
        x = self.decoder(x)

        return x[:, :, 1:29, 1:29], mu, log_var

    def get_latent(self, imgs, labels):
        x = self.encoder(imgs)
        x = x.view(x.size(0), -1)
        one_hots = one_hot(labels, CVAEMNIST.N_CLASSES).float()
        x = torch.cat([x, one_hots], axis=1)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self._sampling(mu, log_var)
        return z

    def generate(self, z, label):
        one_hots = one_hot(label, CVAEMNIST.N_CLASSES).float()
        x = torch.cat([z, one_hots], axis=1)
        x = self.upsample(x)
        x = x.view(x.size(0), 64, 7, 7)
        x = self.decoder(x)
        return x[:, :, 1:29, 1:29]

    def _sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


if __name__ == "__main__":
    from torchsummary import summary

    imgs = torch.rand(3, 1, 28, 28)
    labels = torch.rand(3)
    model = VAEMNIST(2).eval()
    rec, mu, log_var = model(imgs)

    print(rec.size())
    print(mu.size())
    print(log_var.size())

    imgs = torch.rand(3, 1, 28, 28)
    labels = torch.rand(3).long()
    model = CVAEMNIST(2).eval()
    rec, mu, log_var = model(imgs, labels)

    print(rec.size())
    print(mu.size())
    print(log_var.size())
