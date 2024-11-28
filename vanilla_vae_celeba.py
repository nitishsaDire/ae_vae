import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, sampler

# Hyperparameters
RANDOM_SEED = 49
LEARNING_RATE = 0.005
BATCH_SIZE = 64
NUM_EPOCHS = 20
device='cuda'

class VanillaVAE(nn.Module):
    def __init__(self, hidden_dims=2048, latent_dims=128) -> None:
        super(VanillaVAE, self).__init__()
        self.hidden_dims=hidden_dims
        self.latent_dims=latent_dims
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(128, 256, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256, 512, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.Flatten(),
                # nn.Linear(2048, self.hidden_dims)
        )
        
        self.decoder = nn.Sequential(
                    nn.Unflatten(1, (512, 2, 2)),
                    nn.ConvTranspose2d(512, 256, stride=2,  kernel_size=3, padding=1, output_padding=1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(256, 128, stride=2, kernel_size=3, padding=1, output_padding=1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(128, 64, stride=2, kernel_size=3, padding=1, output_padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding=1, output_padding=1),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(32, 32, stride=2, kernel_size=3, padding=1, output_padding=1),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),
                    nn.Tanh(),
            )

        self.decoder_input = nn.Linear(self.latent_dims, self.hidden_dims)
        self.fc_mu = nn.Linear(self.hidden_dims, self.latent_dims)
        # self.bn_mu=nn.BatchNorm1d(self.latent_dims)
        self.fc_var = nn.Linear(self.hidden_dims, self.latent_dims)
        self.bn_var=nn.BatchNorm1d(self.latent_dims)

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
        mu = self.fc_mu(y)
        log_var = self.bn_var(self.fc_var(y))
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


def get_val_loss(valid_loader, model):
    gc.collect()
    model.eval()
    total_loss=total_r_loss=total_kld_loss=0.0
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(valid_loader):
            x=x.to(device)
            y, mu, log_var = model(x)
            loss = loss_function(x,y,mu,log_var)
            total_r_loss+=loss['Reconstruction_Loss']
            total_kld_loss+=loss['KLD']
            total_loss+=loss['loss']
    model.train()
    gc.collect()
    return total_loss/len(valid_loader), total_r_loss/len(valid_loader), total_kld_loss/len(valid_loader)


def train_vae(num_epochs, model, optimizer,
                            train_loader,
                            valid_loader,
                            logging_interval=100,
                            skip_epoch_stats=False,
                            save_model=None):
        
    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}

    start_time = time.time()
    best_val_loss=10000000.0
    running_training_loss=[0,0,0]
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (x, _) in enumerate(train_loader):
            x=x.to(device)
            optimizer.zero_grad()
            
            y, mu, log_var = model(x)
            loss = loss_function(x,y,mu,log_var)
            running_training_loss[0]+=loss['loss']
            running_training_loss[1]+=loss['Reconstruction_Loss']
            running_training_loss[2]+=loss['KLD']

            loss=loss['loss']
            loss.backward()

            optimizer.step()

            log_dict['train_loss_per_batch'].append(loss.item())
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f, recons_loss: %.4f, kld loss: %.4f'
                    % (epoch+1, num_epochs, batch_idx,
                        len(train_loader), running_training_loss[0]/logging_interval, running_training_loss[1]/logging_interval, running_training_loss[2]/logging_interval))
                running_training_loss=[0,0,0]
        val_loss, val_r_loss, val_kld_loss=get_val_loss(valid_loader, model)
        print('Val Loss: %.4f, recons_loss: %.4f, kld loss: %.4f'
            % (val_loss, val_r_loss, val_kld_loss))
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            print("saving model")
            torch.save(model.state_dict(), save_model+f"_{epoch}.pth")

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    return log_dict


def get_dataloaders_celeba(batch_size, num_workers=0):
    if train_transforms is None:
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(148),
                                              transforms.Resize(64),
                                              transforms.ToTensor(),])
    if test_transforms is None:
        test_transforms = transforms.Compose([
                                            transforms.CenterCrop(148),
                                            transforms.Resize(64),
                                            transforms.ToTensor(),])
    train_dataset = datasets.celeba.CelebA(root='/content/VAE/models',
                                            transform=train_transforms, split='train')
    valid_dataset = datasets.celeba.CelebA(root='/content/VAE/models',
                                            transform=test_transforms, split='valid')
    test_dataset = datasets.celeba.CelebA(root='/content/VAE/models',
                                            transform=test_transforms, split='test')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    
    model=VanillaVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, valid_loader, test_loader = get_dataloaders_celeba(batch_size=BATCH_SIZE, num_workers=1)

    log_dict = train_vae(num_epochs=NUM_EPOCHS, model=model,
                                    optimizer=optimizer,
                                    train_loader=train_loader,
                                    valid_loader=valid_loader,
                                    skip_epoch_stats=True,
                                    logging_interval=250,
                                    save_model='celeba_vae')
    
