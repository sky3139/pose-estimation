import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from icecream import ic as print
from functorch import jacrev, vmap
from utils import *

class AE(nn.Module):
    def __init__(self, in_features, hidden1=64, hidden2=128, hidden3=256, hidden4=512):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.LeakyReLU(),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(),
            nn.Linear(hidden2, hidden3),
            nn.LeakyReLU(),
            nn.Linear(hidden3, hidden4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden4, hidden3),
            nn.LeakyReLU(),
            nn.Linear(hidden3, hidden2),
            nn.LeakyReLU(),
            nn.Linear(hidden2, hidden1),
            nn.LeakyReLU(),
            nn.Linear(hidden1, in_features),
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return z, out

class VAE(nn.Module):
    def __init__(self, in_features, hidden1=32, hidden2=64, hidden3=64, latent_dim=64):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.LeakyReLU(),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(),
            nn.Linear(hidden2, hidden3),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(hidden3, latent_dim)
        self.fc_var = nn.Linear(hidden3, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden3),
            nn.LeakyReLU(),
            nn.Linear(hidden3, hidden2),
            nn.LeakyReLU(),
            nn.Linear(hidden2, hidden1),
            nn.LeakyReLU(),
            nn.Linear(hidden1, in_features),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logvar = self.fc_var(result)
        return [mu, logvar]
    
    def decode(self, l):
        return self.decoder(l)

    def decode_distribution(self, l_mu, l_var):
        ###############################################
        # udpate mu
        ###############################################
        x_mu = self.decoder(l_mu)

        ###############################################
        # update covariance
        ###############################################
        # calculate jacobian matrix
        J = vmap(jacrev(self.decoder))(l_mu)
        # covariance matrix of x
        x_var = torch.matmul(torch.matmul(J, l_var), torch.transpose(J, -2, -1))

        return x_mu, x_var
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return [z, out, mu, logvar]

class Transition(nn.Module):
    def __init__(self, in_features, hidden1=64, hidden2=64, hidden3=64):
        super(Transition, self).__init__()
        self.in_features = in_features
        self.residual = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.LeakyReLU(),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(),
            nn.Linear(hidden2, hidden3),
            nn.LeakyReLU(),
            nn.Linear(hidden3, in_features),
        )
        self.l_noise = nn.Parameter(torch.tensor([-1.])) # initialize to 0.2689

    def transition(self, mu):
        return mu + self.residual(mu)

    def forward(self, l_mu, l_var):
        ###############################################
        # udpate mu
        ###############################################
        new_mu = self.transition(l_mu)

        ###############################################
        # update covariance
        ###############################################
        # calculate jacobian matrix
        # from torch.autograd.functional import jacobian
        # J = jacobian(self.transition, l_mu, create_graph=True, vectorize=True)
        # J = torch.stack([J[batch_idx, :, batch_idx, :] for batch_idx in range(l_mu.shape[0])], dim=0)      
        J = vmap(jacrev(self.transition))(l_mu)
        # covariance matrix of l t+1
        new_var = torch.matmul(torch.matmul(J, l_var), J.permute(0, 2, 1))
        new_var += (self.l_noise.sigmoid()**2)*torch.eye(new_var.shape[-1]).to(new_var.device)

        return new_mu, new_var


if __name__ == '__main__':
    ae = AE(in_features=64)
    x = torch.randn(16, 64)
    z, out = ae(x)
    # print(z.shape, out.shape)
    transition = Transition(in_features=64)
    mu = torch.randn(16, 64, requires_grad=True)
    logvar = torch.randn(16, 64, requires_grad=True)
    # print(mu, logvar)
    out, new_mu, new_logvar = transition(mu, logvar)
    # print(new_mu, new_logvar)
