import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparameters
RANDOM_SEED = 49
LEARNING_RATE = 0.0005
BATCH_SIZE = 256
NUM_EPOCHS = 30
NUM_CLASSES = 10
device='cuda'

class VanillaAE(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(VanillaAE, self).__init__()
        
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.Flatten(),
                nn.Linear(3136, 2)
        )
        
        self.decoder = nn.Sequential(
                torch.nn.Linear(2, 3136),
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


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder(num_epochs, model, optimizer, train_loader, save_model=None):
        
    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}
    loss_fn = F.mse_loss
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (x, _) in enumerate(train_loader):
            x=x.to(device)
            y = model(x)
            loss = loss_fn(y, x)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            log_dict['train_loss_per_batch'].append(loss.item())
            if not batch_idx % 100:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'% (epoch+1, num_epochs, batch_idx, len(train_loader), loss))

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
    
    model=VanillaAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, valid_loader, test_loader = get_dataloaders_mnist(batch_size=BATCH_SIZE)

    log_dict = train_autoencoder(num_epochs=NUM_EPOCHS, model=model,
                                    optimizer=optimizer,
                                    train_loader=train_loader,
                                    skip_epoch_stats=True,
                                    logging_interval=250,
                                    save_model='ae.pth')
    
