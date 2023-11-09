import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

# custom VAE implementation
class VAE_encoder(nn.Module):
    def __init__(self,input_dim:int, 
                 hidden_dims:List[int], 
                 latent_dim:int, 
                 hidden_activation:torch.nn, 
                 output_mu_activation:torch.nn,
                 output_logvar_activation:torch.nn):
        
        super().__init__()

        layers=[]
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(hidden_activation)
            # update input_dim for next layer
            input_dim = h_dim
        
        self.layers = nn.Sequential(*layers)

        self.output_mu = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
            output_mu_activation
        )

        self.output_logvar = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
            output_logvar_activation
        )
    
    def forward(self, x ):
        x = self.layers(x) 
        mu = self.output_mu(x) # size [Batch size x latent_dim]
        logvar = self.output_logvar(x) # size [Batch size x latent_dim]
        return mu, logvar
        
        
class VAE_decoder(nn.Module):
    def __init__(self,input_dim:int, 
                 hidden_dims:List[int], 
                 output_dim:int, 
                 hidden_activation:torch.nn, 
                 output_mu_activation:torch.nn,
                 output_logvar_activation:torch.nn):
        super().__init__()

        layers=[]
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(hidden_activation)
            # update input_dim for next layer
            input_dim = h_dim

        self.layers = nn.Sequential(*layers)

        self.output_mu = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            output_mu_activation
        )

        self.output_logvar = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            output_logvar_activation
        )
    
    def forward(self, z):
        z = self.layers(z)
        mu = self.output_mu(z) # size [Batch size x output_dim]
        logvar = self.output_logvar(z) # size [Batch size x output_dim]
        return mu, logvar
        

class VAE(nn.Module):
    def __init__(self, input_dim:int, 
                 hidden_dims:List[int], 
                 latent_dim:int, 
                 hidden_activation:torch.nn, 
                 encoder_output_mu_activation:torch.nn,
                 encoder_output_logvar_activation:torch.nn,
                 decoder_output_mu_activation:torch.nn,
                 decoder_output_logvar_activation:torch.nn):
        super().__init__()

        self.encoder = VAE_encoder(input_dim, 
                                   hidden_dims, 
                                   latent_dim, 
                                   hidden_activation, 
                                   encoder_output_mu_activation,
                                   encoder_output_logvar_activation)
        
        self.decoder = VAE_decoder(latent_dim, 
                                   hidden_dims[::-1],  # reverse hidden dims (case where the encoder & decoder are symetric)
                                   input_dim, 
                                   hidden_activation, 
                                   decoder_output_mu_activation,
                                   decoder_output_logvar_activation)
        
    def encode(self, x):
        mu_enc,logvar_enc = self.encoder(x)
        return mu_enc,logvar_enc
    
    def decode(self, z):
        mu_dec,logvar_dec = self.decoder(z)
        return mu_dec,logvar_dec
    

    @staticmethod
    def reparametrization_trick(mu, log_var):
        epsilon = torch.randn_like(mu)  # the Gaussian random noise with shape of mu
        # as we are working with log(sigma^2)=2log(sigma), in order to have sigma with need to do
        # e^(0.5*ln(sigma^2)) which is equal to sigma
        return mu + torch.exp(0.5 * log_var) * epsilon
    
    def forward(self, x):
        mu_z, log_var_z = self.encode(x)
        # get z by sampling from the normal distribution with mean mu_z and variance log_var_z
        z_rep = self.reparametrization_trick(mu_z, log_var_z)
        # get mu_x and log_var_x from z_rep
        mu_x, log_var_x = self.decode(z_rep)
        return mu_x, log_var_x, z_rep, mu_z, log_var_z

# Computes the objective function of the VAE
def VAE_loss(x, mu_x, log_var_x, mu_z, log_var_z, r=1.0):
    D = mu_x.shape[1] # input space dimension
    d = mu_z.shape[1] # latent space dimension 

    # P(X|Z) - The Probabilistic Decoder: 
    # This term represents the likelihood of observing X given the latent variable Z
    if log_var_x.shape[1] == 1: 
        P_X_Z = + 0.5 * (D * log_var_x + (((x - mu_x) ** 2) / log_var_x.exp()).sum(dim=1, keepdim=True)).mean()
    else:
        P_X_Z = + 0.5 * (log_var_x.sum(dim=1, keepdim=True)
                         + (((x - mu_x) ** 2) / log_var_x.exp()).sum(dim=1, keepdim=True)).mean()

    # Q(Z|X) - The Probabilistic Encoder: 
    # This term reflects the entropy of the approximate posterior distribution which is modeled by the encoder. 
    if log_var_z.shape[1] == 1: 
        Q_Z_X = - 0.5 * (d * log_var_z).mean()
    else:
        Q_Z_X = - 0.5 * log_var_z.sum(dim=1, keepdim=True).mean()


    # P(Z) - The Prior: 
    # This term enforces the latent variables to follow a certain prior distribution,
    # which is here a standard Gaussian distribution:
    if log_var_z.shape[1] == 1:
        P_Z = + 0.5 * ((mu_z ** 2).sum(dim=1, keepdim=True) + d * log_var_z.exp()).mean()
    else:
        P_Z = + 0.5 * ((mu_z ** 2).sum(dim=1, keepdim=True) + log_var_z.exp().sum(dim=1, keepdim=True)).mean()
    # it is combined with a scaling factor r (which adjust the influence of the latent 
    # space regularization during training)

    return P_X_Z + r * Q_Z_X + r * P_Z
