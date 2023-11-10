import torch.nn as nn
import torch


# used to save the whole model as .pt
class VAE_RBF(nn.Module):
    def __init__(self,VAE_model,RBF_model):
        super().__init__()

        self.VAE = VAE_model
        self.RBF = RBF_model

    def forward(self, x ):
        MU_X_eval, _, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval = self.VAE(x)
        LOG_VAR_X_RBF = self.RBF(Z_ENC_eval)
        return MU_X_eval, LOG_VAR_X_RBF, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval
    
    def forward_RBF(self, z):
        LOG_VAR_X_RBF = self.RBF(z)
        return LOG_VAR_X_RBF
    
    def forward_VAE(self, x):
        return self.VAE(x)