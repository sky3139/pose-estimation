import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import os
# from icecream import ic as print
from datetime import datetime

from models import Transition, VAE
from dataset import Dataset_Transition
from utils import *


# Path setting
data_path = "./data/human"
model_path = "./saved/models/transition-noise"
sum_path = os.path.join(model_path, "runs_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))
vae_checkpoint = "./saved/models/vae-noise/ae_best.pth"
model_checkpoint = None#'./saved/models/transition-test/transition_epoch050.pth'

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
trans_in_features = 64
vae_in_features = 25
num_joints = 12
num_pos = num_joints*3
num_uv = num_joints*2
epochs = 100
batch_size = 16
learning_rate = 5e-4
uv_weight = 5e-2
pos_weight = 5e-2
kl_weight = 5e-4
pred_frames = 2
log_interval = 500
# z_var = torch.diag_embed(torch.tensor([74.55658241, 82.26965012, 68.03699526, 90.30622388,
#                                        277.74164245, 69.31363767, 388.1904017, 81.04009582,
#                                        69.03923907, 85.00237238, 124.95977423, 114.05540344,
#                                        120.50384591, 125.69350324, 141.93116438, 145.27943616,
#                                        53.51215166, 94.8005174, 41.2241692, 108.58219488,
#                                        47.47740765, 112.09194289, 61.31272559, 115.42536559])).to(device)
# l_noise = (0.2**2)*torch.eye(trans_in_features).to(device)
pos_noise = (0.02**2)*torch.eye(num_pos).to(device)
uv_noise = (10**2)*torch.eye(num_uv).to(device)

def loss_func(model, criterion, x, r, intrinsics, uv_1, uv_2, uv_3, uv_4, uv_5, pos_1, pos_2, pos_3, pos_4, pos_5, x_1, x_2, x_3, x_4, x_5):
    batch_size = x.shape[0]
    with torch.no_grad():
        l_mu, l_logvar = vae.encode(x)
        l_var = torch.diag_embed(torch.exp(l_logvar))
    uv_list = [uv_1, uv_2, uv_3, uv_4, uv_5]
    pos_list = [pos_1, pos_2, pos_3, pos_4, pos_5]
    x_list = [x_1, x_2, x_3, x_4, x_5]
    loss = 0
    loss_kld_all = 0
    loss_2d_all = 0
    loss_3d_all = 0
    for t in range(pred_frames):
        # one step forward
        l_mu, l_var = model(l_mu, l_var)
        # KL Loss
        with torch.no_grad():
            l_mu_next, l_logvar_next = vae.encode(x_list[t])
            l_var_next = torch.diag_embed(torch.exp(l_logvar_next))
        p = MultivariateNormal(l_mu, l_var)
        q = MultivariateNormal(l_mu_next, l_var_next)
        loss_kld = kl_divergence(p, q).mean()
        loss += kl_weight * loss_kld
        loss_kld_all += loss_kld

        x_mu, x_var = vae.decode_distribution(l_mu, l_var)
        # 3D Loss
        pos_mu, pos_var = x2pos_distribution(x_mu, x_var, r)
        # loss_3d = criterion(pos_mu, pos_list[t])   # deterministic version
        # loss_3d = criterion(pos_mu, tensor2skeleton(x_list[t], r))   # deterministic version
        pos_u, pos_s, pos_v = torch.svd(pos_var + pos_noise)
        loss_3d = torch.log(pos_s).sum(-1).mean() + torch.matmul(torch.matmul((pos_list[t] - pos_mu).reshape(batch_size, 1, -1), torch.inverse(pos_var + pos_noise)), (pos_list[t] - pos_mu).reshape(batch_size, -1, 1)).squeeze().mean()
        # print(0.5*loss_3d+ 32*torch.log(torch.tensor([6.28]).to(device)))
        # print(torch.eig((pos_var + pos_noise).squeeze()))
        # print(criterion(pos_list[t], tensor2skeleton(x, r)))
        # print(-MultivariateNormal(pos_mu.reshape(batch_size, -1), covariance_matrix=pos_var+pos_noise).log_prob(pos_list[t].reshape(batch_size, -1)).mean())
        # loss_3d = -MultivariateNormal(pos_mu.reshape(batch_size, -1), covariance_matrix=pos_var+pos_noise).log_prob(pos_list[t].reshape(batch_size, -1)).mean()
        loss += pos_weight * loss_3d
        loss_3d_all += loss_3d

        # 2D Loss
        uv_mu, uv_var = pos2uv_distribution(pos_mu, pos_var, intrinsics)
        # print((uv_list[t], pos2uv(tensor2skeleton(x_list[t], r), intrinsics)))
        # loss_2d = criterion(uv_mu, uv_list[t])   # deterministic version
        uv_u, uv_s, uv_v = torch.svd(uv_var + uv_noise)
        loss_2d = torch.log(uv_s).sum(-1).mean() + torch.matmul(torch.matmul((uv_list[t] - uv_mu).reshape(batch_size, 1, -1), torch.inverse(uv_var + uv_noise)), (uv_list[t] - uv_mu).reshape(batch_size, -1, 1)).squeeze().mean()
        # print(torch.eig((uv_var+uv_noise).squeeze()))
        # loss_2d = -MultivariateNormal(uv_mu.reshape(batch_size, -1), covariance_matrix=uv_var+uv_noise, validate_args=False).log_prob(uv_list[t].reshape(batch_size, -1)).mean()
        loss += uv_weight * loss_2d
        loss_2d_all += loss_2d
        # print(loss, loss_kld, uv_weight * loss_2d, pos_weight * loss_3d)

    return loss, loss_kld_all, loss_3d_all, loss_2d_all

if __name__ == '__main__':
    # Load data
    train_set = Dataset_Transition(data_path=data_path, mode='train')
    val_set = Dataset_Transition(data_path=data_path, mode='validation')
    test_set = Dataset_Transition(data_path=data_path, mode='test')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    # Create transition model
    model = Transition(in_features=trans_in_features).to(device)
    # Load checkpoint
    if model_checkpoint is not None:
        model.load_state_dict(torch.load(model_checkpoint))
        print('Model Checkpoint Loaded'.center(60, '#'))

    # Load vae model
    vae = VAE(in_features=vae_in_features).to(device)
    vae.eval()
    # Load checkpoint
    if vae_checkpoint is not None:
        vae.load_state_dict(torch.load(vae_checkpoint))
        print('VAE Checkpoint Loaded'.center(60, '#'))

    # Create loss criterion & optimizer & tensorboard writer
    criterion = nn.SmoothL1Loss() #nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    for epoch in range(epochs):
        #################################################
        # Train the model
        #################################################
        model.train()
        losses = []
        losses_kld = []
        losses_3d = []
        losses_2d = []
        for batch_idx, data in enumerate(train_loader):
            # get the inputs
            x, r, intrinsics, uv_1, uv_2, uv_3, uv_4, uv_5, pos_1, pos_2, pos_3, pos_4, pos_5, x_1, x_2, x_3, x_4, x_5 = [item.to(device) for item in data]

            optimizer.zero_grad()
            loss, loss_kld, loss_3d, loss_2d = loss_func(model, criterion, x, r, intrinsics, uv_1, uv_2, uv_3, uv_4, uv_5, pos_1, pos_2, pos_3, pos_4, pos_5, x_1, x_2, x_3, x_4, x_5)
            losses.append(loss.item())
            losses_kld.append(loss_kld.item())
            losses_3d.append(loss_3d.item())
            losses_2d.append(loss_2d.item())

            # backward & optimize
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % log_interval == 0:
                print("epoch {:3d} | iteration {:5d} | Loss {:.6f}".format(epoch+1, batch_idx+1, loss.item()))

        # Compute the average loss
        training_loss = sum(losses)/len(losses)
        loss_kld = sum(losses_kld)/len(losses_kld)
        loss_3d = sum(losses_3d)/len(losses_3d)
        loss_2d = sum(losses_2d)/len(losses_2d)
        print("Average Training Loss of Epoch {}: {:.6f} | KLD {:.6f} | 3D {:.6f} | 2D {:.6f}".format(epoch+1, training_loss, loss_kld, loss_3d, loss_2d))

        
        #################################################
        # Validate the model
        #################################################
        model.eval()
        losses = []
        # with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            # get the inputs
            x, r, intrinsics, uv_1, uv_2, uv_3, uv_4, uv_5, pos_1, pos_2, pos_3, pos_4, pos_5, x_1, x_2, x_3, x_4, x_5 = [item.to(device) for item in data]
            loss, loss_kld, loss_3d, loss_2d = loss_func(model, criterion, x, r, intrinsics, uv_1, uv_2, uv_3, uv_4, uv_5, pos_1, pos_2, pos_3, pos_4, pos_5, x_1, x_2, x_3, x_4, x_5)
            losses.append(loss.item())

        # Compute the average loss
        val_loss = sum(losses)/len(losses)
        print("Average Val Loss: {:.6f}".format(val_loss))

        #################################################
        # Test the model
        #################################################
        model.eval()
        losses = []
        # with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # get the inputs
            x, r, intrinsics, uv_1, uv_2, uv_3, uv_4, uv_5, pos_1, pos_2, pos_3, pos_4, pos_5, x_1, x_2, x_3, x_4, x_5 = [item.to(device) for item in data]
            loss, loss_kld, loss_3d, loss_2d = loss_func(model, criterion, x, r, intrinsics, uv_1, uv_2, uv_3, uv_4, uv_5, pos_1, pos_2, pos_3, pos_4, pos_5, x_1, x_2, x_3, x_4, x_5)
            losses.append(loss.item())

        # Compute the average loss
        test_loss = sum(losses)/len(losses)
        print("Average Test Loss: {:.6f}".format(test_loss))
        # writer.add_scalars('Loss', {'test': test_loss}, epoch+1)

        # Save model
        torch.save(model.state_dict(), os.path.join(model_path, "transition_epoch{:03d}.pth".format(epoch+1)))
        print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_path, "transition_best.pth"))
            print("Best Model Saved".center(60, '#'))
