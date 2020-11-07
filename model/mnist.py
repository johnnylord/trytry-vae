import torch
import torch.nn as nn
from torch.nn.functional import one_hot


class VAEMNIST(nn.Module):
    """Implementation of VAE model on mnist dataset"""
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
                            nn.Linear(784, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 32),
                            nn.ReLU(inplace=True))

        self.mu = nn.Linear(32, latent_dim)
        self.log_var = nn.Linear(32, latent_dim)
        self.upsample = nn.Linear(latent_dim, 32)

        self.decoder = nn.Sequential(
                            nn.Linear(32, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 784),
                            nn.Sigmoid())

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)

        # Encoder
        x = self.encoder(x)

        # Latent
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self._sampling(mu, log_var)

        # Decoder
        x = self.upsample(z)
        x = self.decoder(x)

        x = x.view(x.size(0), 1, 28, 28)
        return x, mu, log_var

    def get_latent(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self._sampling(mu, log_var)
        return z

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
                            nn.Linear(784+CVAEMNIST.N_CLASSES, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 32),
                            nn.ReLU(inplace=True))

        self.mu = nn.Linear(32, latent_dim)
        self.log_var = nn.Linear(32, latent_dim)
        self.upsample = nn.Linear(latent_dim+CVAEMNIST.N_CLASSES, 32)

        self.decoder = nn.Sequential(
                            nn.Linear(32, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 784),
                            nn.Sigmoid())

    def forward(self, imgs, labels):
        one_hots = one_hot(labels, CVAEMNIST.N_CLASSES).float()
        x = torch.cat([imgs.view(imgs.size(0), -1), one_hots], axis=1)
        x = self.encoder(x)

        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self._sampling(mu, log_var)

        x = torch.cat([z, one_hots], axis=1)
        x = self.upsample(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x, mu, log_var

    def get_latent(self, imgs, labels):
        one_hots = one_hot(labels, CVAEMNIST.N_CLASSES).float()
        x = torch.cat([imgs.view(imgs.size(0), -1), one_hots], axis=1)
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self._sampling(mu, log_var)
        return z

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
