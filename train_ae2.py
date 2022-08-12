import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from models import AE
from dataset import Dataset_HUMAN

# Path setting
data_path = "./data/human"
model_path = "./saved/models/ae"

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
in_features = 25
epochs = 100
batch_size = 16
learning_rate = 1e-5 ##动态调学习率会不会好一点
log_interval = 100

if __name__ == '__main__':
    # Load data
    train_set = Dataset_HUMAN(data_path=data_path, mode='train')
    val_set = Dataset_HUMAN(data_path=data_path, mode='validation')
    test_set = Dataset_HUMAN(data_path=data_path, mode='test')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    # Create model
    model = AE(in_features=in_features).to(device)

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
        for batch_idx, data in enumerate(train_loader):
            # get the inputs
            x = data.to(device)

            optimizer.zero_grad()
            # forward
            z, out = model(x)

            # compute the loss
            rec_loss = rec_criterion(out, x)
            reg_loss = torch.mean(torch.norm(z, dim=-1).pow(2))
            loss = rec_loss + reg_loss
            rec_losses.append(rec_loss.item())
            reg_losses.append(reg_loss.item())
            losses.append(loss.item())

            # backward & optimize
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % log_interval == 0:
                print("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Rec {:.6f} | Reg {:.6f}".format(epoch+1, batch_idx+1, loss.item(), rec_loss.item(), reg_loss.item()))

        # Compute the average loss
        rec_loss = sum(rec_losses)/len(rec_losses)
        reg_loss = sum(reg_losses)/len(reg_losses)
        training_loss = sum(losses)/len(losses)
        print("Average Training Loss of Epoch {}: {:.6f} | Rec {:.6f} | Reg {:.6f}".format(epoch+1, training_loss, rec_loss, reg_loss))

        
        #################################################
        # Validate the model
        #################################################
        model.eval()
        losses = []
        rec_losses = []
        reg_losses = []
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                # get the inputs
                x = data.to(device)

                # forward
                z, out = model(x)

                # compute the loss
                rec_loss = rec_criterion(out, x)
                reg_loss = torch.mean(torch.norm(z, dim=-1).pow(2))
                loss = rec_loss + reg_loss
                rec_losses.append(rec_loss.item())
                reg_losses.append(reg_loss.item())
                losses.append(loss.item())

        # Compute the average loss
        rec_loss = sum(rec_losses)/len(rec_losses)
        reg_loss = sum(reg_losses)/len(reg_losses)
        val_loss = sum(losses)/len(losses)
        print("Average Val Loss: {:.6f} | Rec {:.6f} | Reg {:.6f}".format(val_loss, rec_loss, reg_loss))

        #################################################
        # Test the model
        #################################################
        model.eval()
        losses = []
        rec_losses = []
        reg_losses = []
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                # get the inputs
                x = data.to(device)

                # forward
                z, out = model(x)

                # compute the loss
                rec_loss = rec_criterion(out, x)
                reg_loss = torch.mean(torch.norm(z, dim=-1).pow(2))
                loss = rec_loss + reg_loss
                rec_losses.append(rec_loss.item())
                reg_losses.append(reg_loss.item())
                losses.append(loss.item())

        # Compute the average loss
        rec_loss = sum(rec_losses)/len(rec_losses)
        reg_loss = sum(reg_losses)/len(reg_losses)
        test_loss = sum(losses)/len(losses)
        print("Average Test Loss: {:.6f} | Rec {:.6f} | Reg {:.6f}".format(test_loss, rec_loss, reg_loss))

        # Save model
        torch.save(model.state_dict(), os.path.join(model_path, "ae_epoch{:03d}.pth".format(epoch+1)))
        print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_path, "ae_best.pth"))
