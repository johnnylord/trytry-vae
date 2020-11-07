import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST
from tensorboardX import SummaryWriter

from model.mnist import VAEMNIST, CVAEMNIST

__all__ = [ "MNISTAgent" ]

class MNISTAgent:

    def __init__(self, config):
        self.config = config

        # Determine environment
        self.device = config["train"]["device"] if torch.cuda.is_available() else "cpu"

        # Data Preprocessing
        tr_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
            ])
        te_transform = T.Compose([ T.ToTensor(), ])

        # Dataset
        train_dataset = MNIST(config['dataset']['root'],
                            transform=tr_transform, train=True, download=True)
        test_dataset = MNIST(config['dataset']['root'],
                            transform=te_transform, train=False, download=True)

        # Dataloader
        self.train_loader = DataLoader(train_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=config['dataloader']['num_workers'],
                                shuffle=True)
        self.test_loader = DataLoader(test_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=config['dataloader']['num_workers'],
                                shuffle=False)

        # Model
        if config['train']['mode'] == 'cvae':
            model = CVAEMNIST(latent_dim=config['model']['latent_size'])
        else:
            model = VAEMNIST(latent_dim=config['model']['latent_size'])
        model = model.to(self.device)
        self.model = model.train()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config['optim']['lr'])

        # Objective function
        self.rec_loss = nn.BCELoss(reduction="sum")
        self.reg_loss = lambda mu, log_var: 0.5 * torch.sum(log_var.exp() + mu.pow(2) - 1 - log_var)

        # Tensorboard
        log_dir = osp.join(config['train']['log_dir'], config['train']['exp_name'])
        self.writer = SummaryWriter(log_dir)

        # Running state
        self.current_epoch = -1
        self.current_loss = 100000

        # Resume training
        # Easy to train, don't need to resume training feature

    def train(self):
        for epoch in range(self.config['train']['n_epochs']):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.validation()

    def train_one_epoch(self):
        running_loss = 0

        self.model.train()
        for batch_idx, (imgs, labels) in enumerate(self.train_loader):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            if self.config['train']['mode'] == 'cvae':
                rec_imgs, mu, var = self.model(imgs, labels)
            else:
                rec_imgs, mu, var = self.model(imgs)
            rec_loss = self.rec_loss(rec_imgs, imgs) / len(imgs)
            reg_loss = self.reg_loss(mu, var) / len(imgs)
            loss = rec_loss + reg_loss
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()*len(imgs)

            if batch_idx % self.config['train']['log_interval'] == 0:
                print(("[Epoch {}:{}], Step: {}%, Total Loss: {:.2f}, "
                        "Rec Loss: {:.2f}, Reg Loss: {:.2f}").format(
                            self.current_epoch, self.config['train']['n_epochs'],
                            int(batch_idx/len(self.train_loader)*100), loss,
                            rec_loss, reg_loss
                            ))

        epoch_loss = running_loss / len(self.train_loader.dataset)
        print("[Epoch {}:{}], Train Loss: {:.2f}".format(
            self.current_epoch, self.config['train']['n_epochs'], epoch_loss))
        self.writer.add_scalar("Train Loss", epoch_loss, self.current_epoch)

    def validation(self):
        running_loss = 0

        self.model.eval()
        for batch_idx, (imgs, labels) in enumerate(self.test_loader):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            if self.config['train']['mode'] == 'cvae':
                rec_imgs, mu, var = self.model(imgs, labels)
            else:
                rec_imgs, mu, var = self.model(imgs)
            rec_loss = self.rec_loss(rec_imgs, imgs) / len(imgs)
            reg_loss = self.reg_loss(mu, var) / len(imgs)
            loss = (rec_loss + reg_loss)
            running_loss += loss.item()*len(imgs)

        epoch_loss = running_loss / len(self.test_loader.dataset)
        print("[Epoch {}:{}], Test Loss: {:.2f}".format(
            self.current_epoch, self.config['train']['n_epochs'], epoch_loss))
        self.writer.add_scalar("Test Loss", epoch_loss, self.current_epoch)

        if self.current_loss > epoch_loss:
            self.current_loss = epoch_loss
            self._save_checkpoint()

    def finalize(self):
        pass

    def _save_checkpoint(self):
        checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'current_epoch': self.current_epoch,
                'current_loss': self.current_loss
                }
        checkpoint_dir = osp.join(self.config['train']['log_dir'],
                                "{}_checkpoint".format(self.config['train']['exp_name']))
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        torch.save(checkpoint, osp.join(checkpoint_dir, 'best.pth'))
        print("Save best checkpoint to '{}'".format(osp.join(checkpoint_dir, 'best.pth')))
