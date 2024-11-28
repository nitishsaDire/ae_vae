import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparameters
RANDOM_SEED = 49
LEARNING_RATE = 0.005
BATCH_SIZE = 256
NUM_EPOCHS = 30
NUM_CLASSES = 10
device='cuda'

class VanillaVAE(nn.Module):
    def __init__(self, hidden_dims=32, latent_dims=2) -> None:
        super(VanillaVAE, self).__init__()
        self.hidden_dims=hidden_dims
        self.latent_dims=latent_dims
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.Flatten(),
                nn.Linear(3136, self.hidden_dims)
        )
        
        self.decoder = nn.Sequential(
                torch.nn.Linear(self.hidden_dims, 3136),
                nn.Unflatten(1, (64, 7, 7)),
                nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0),
                nn.Sigmoid()
        )

        self.decoder_input = nn.Linear(self.latent_dims, self.hidden_dims)
        self.fc_mu = nn.Linear(self.hidden_dims, self.latent_dims)
        self.fc_var = nn.Linear(self.hidden_dims, self.latent_dims)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        return result

    def encode(self, input):
        y = self.encoder(input)
        mu, log_var = self.fc_mu(y), self.fc_var(y)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def forward(self, input, **kwargs):
        z, mu, log_var=self.encode(input)
        return  [self.decode(z), mu, log_var]


def loss_function(x, y, mu, log_var, kld_weight=0.00025):
    recons_loss =F.mse_loss(x, y)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    loss = recons_loss + kld_weight * kld_loss
    return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


def train_autoencoder(num_epochs, model, optimizer,
                            train_loader,
                            logging_interval=100,
                            skip_epoch_stats=False,
                            save_model=None):
        
    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (x, _) in enumerate(train_loader):
            x=x.to(device)

            y, mu, log_var = model(x)
            loss = loss_function(x,y,mu,log_var)
            recons_loss=loss['Reconstruction_Loss']
            kld_loss=loss['KLD']
            loss=loss['loss']
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            log_dict['train_loss_per_batch'].append(loss.item())
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f, recons_loss: %.4f, kld loss: %.4f'
                    % (epoch+1, num_epochs, batch_idx,
                        len(train_loader), loss, recons_loss, kld_loss))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    return log_dict


def get_dataloaders_mnist(batch_size):
    tfs = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='data', train=True, transform=tfs,download=True)
    valid_dataset = datasets.MNIST(root='data', train=True, transform=tfs)
    test_dataset = datasets.MNIST(root='data', train=False, transform=tfs)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    
    model=VanillaVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, valid_loader, test_loader = get_dataloaders_mnist(batch_size=BATCH_SIZE, num_workers=1)

    log_dict = train_autoencoder(num_epochs=NUM_EPOCHS, model=model,
                                    optimizer=optimizer,
                                    train_loader=train_loader,
                                    skip_epoch_stats=True,
                                    logging_interval=250,
                                    save_model='vae.pth')
    
