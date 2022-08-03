import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from models import VAE
from dataset import Dataset_AE

# Path setting
data_path = "./data/3DPW/sequenceFiles"
model_path = "./saved/models/vae-noise"

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
in_features = 25
epochs = 1000
batch_size = 16
learning_rate = 1e-5
kld_weight = 0.000025
log_interval = 100

if __name__ == '__main__':
    # Load data
    train_set = Dataset_AE(data_path=data_path, mode='train')
    val_set = Dataset_AE(data_path=data_path, mode='validation')
    test_set = Dataset_AE(data_path=data_path, mode='test')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    # Create model
    model = VAE(in_features=in_features).to(device)

    # Create loss criterion & optimizer
    rec_criterion = nn.MSELoss()
    reg_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    for epoch in range(epochs):
        #################################################
        # Train the model
        #################################################
        model.train()
        losses = []
        rec_losses = []
        reg_losses = []
        kld_losses = []
        for batch_idx, data in enumerate(train_loader):
            # get the inputs
            x = data.to(device)

            optimizer.zero_grad()
            # forward
            z, out, mu, log_var = model(x)

            # compute the loss
            rec_loss = rec_criterion(out, x)
            reg_loss = torch.tensor([0]).to(device)# torch.mean(torch.norm(z, dim=-1).pow(2))
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            loss = rec_loss + reg_loss + kld_weight * kld_loss
            rec_losses.append(rec_loss.item())
            reg_losses.append(reg_loss.item())
            kld_losses.append(kld_loss.item())
            losses.append(loss.item())

            # backward & optimize
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % log_interval == 0:
                print("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Rec {:.6f} | Reg {:.6f} | Kld {:.6f}".format(epoch+1, batch_idx+1, loss.item(), rec_loss.item(), reg_loss.item(), kld_loss.item()))

        # Compute the average loss
        rec_loss = sum(rec_losses)/len(rec_losses)
        reg_loss = sum(reg_losses)/len(reg_losses)
        kld_loss = sum(kld_losses)/len(kld_losses)
        training_loss = sum(losses)/len(losses)
        print("Average Training Loss of Epoch {}: {:.6f} | Rec {:.6f} | Reg {:.6f} | Kld {:.6f}".format(epoch+1, training_loss, rec_loss, reg_loss, kld_loss))

        
        #################################################
        # Validate the model
        #################################################
        model.eval()
        losses = []
        rec_losses = []
        reg_losses = []
        kld_losses = []
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                # get the inputs
                x = data.to(device)

                # forward
                z, out, mu, log_var = model(x)

                # compute the loss
                rec_loss = rec_criterion(out, x)
                reg_loss = torch.tensor([0]).to(device)# torch.mean(torch.norm(z, dim=-1).pow(2))
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                loss = rec_loss + reg_loss + kld_weight * kld_loss
                rec_losses.append(rec_loss.item())
                reg_losses.append(reg_loss.item())
                kld_losses.append(kld_loss.item())
                losses.append(loss.item())

        # Compute the average loss
        rec_loss = sum(rec_losses)/len(rec_losses)
        reg_loss = sum(reg_losses)/len(reg_losses)
        kld_loss = sum(kld_losses)/len(kld_losses)
        val_loss = sum(losses)/len(losses)
        print("Average Val Loss: {:.6f} | Rec {:.6f} | Reg {:.6f} | Kld {:.6f}".format(val_loss, rec_loss, reg_loss, kld_loss))

        #################################################
        # Test the model
        #################################################
        model.eval()
        losses = []
        rec_losses = []
        reg_losses = []
        kld_losses = []
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                # get the inputs
                x = data.to(device)

                # forward
                z, out, mu, log_var = model(x)

                # compute the loss
                rec_loss = rec_criterion(out, x)
                reg_loss = torch.tensor([0]).to(device)# torch.mean(torch.norm(z, dim=-1).pow(2))
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                loss = rec_loss + reg_loss + kld_weight * kld_loss
                rec_losses.append(rec_loss.item())
                reg_losses.append(reg_loss.item())
                kld_losses.append(kld_loss.item())
                losses.append(loss.item())

        # Compute the average loss
        rec_loss = sum(rec_losses)/len(rec_losses)
        reg_loss = sum(reg_losses)/len(reg_losses)
        kld_loss = sum(kld_losses)/len(kld_losses)
        test_loss = sum(losses)/len(losses)
        # std = torch.exp(0.5 * log_var)
        # var = torch.diag_embed(std)
        # print(torch.det(var))
        print("Average Test Loss: {:.6f} | Rec {:.6f} | Reg {:.6f} | Kld {:.6f}".format(test_loss, rec_loss, reg_loss, kld_loss))

        # Save model
        torch.save(model.state_dict(), os.path.join(model_path, "ae_epoch{:03d}.pth".format(epoch+1)))
        print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_path, "ae_best.pth"))
            print("Best Model Saved".center(60, '#'))
