import torch.nn as nn
import torch
from typing import List
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

class RBFNN(nn.Module):
    def __init__(self,a,k_rbf,zeta_rbf,W_dim,latent_dim):
        super().__init__()

        self.a = a
        self.k_rbf = k_rbf
        self.zeta_rbf = zeta_rbf # to avoid division by zero
        self.W_rbf = nn.Parameter(torch.rand(W_dim, k_rbf,requires_grad=True))
        self.centers_ids = None # not saved in the model state dict
        # register buffer to save it in the state dict
        self.register_buffer('centers_rbf', torch.empty((k_rbf,latent_dim)))
        self.register_buffer('lambdas_k', torch.empty((self.k_rbf,1)))

    def fit_kmeans(self,latent_means,display=False):
        kmeans_model = KMeans(n_clusters=self.k_rbf, n_init=30, random_state=0,max_iter=1000).fit(latent_means)
        centers_ids = kmeans_model.labels_
        centers_rbf = torch.from_numpy(kmeans_model.cluster_centers_.astype(np.float32))
        self.centers_rbf = centers_rbf
        self.centers_ids = centers_ids # not saved in the state dict
        if(display):
            #plotting the results:
            for i in np.unique(centers_ids):
                plt.scatter(latent_means[centers_ids == i , 0] , latent_means[centers_ids == i , 1] , label = i)
            plt.show()
        return centers_rbf,centers_ids
    
    def compute_bandwidth(self,latent_means):
        lambdas_k = np.zeros((self.k_rbf, 1))
        for k in range(self.k_rbf):
            inds_k = (self.centers_ids == k)
            points = latent_means[inds_k, :]
            c_k = self.centers_rbf[k, :].reshape(-1, 1)
            S = (np.diag((points - c_k.T).T @ (points - c_k.T))/points.shape[0]).min()
            lambdas_k[k, 0] = np.sqrt(S)
        lambdas_k = 1.25 * lambdas_k  # Increase the Sigma by a global factor to smooth the metric a bit, a=1.25
        lambdas_k = 0.5 / (lambdas_k ** 2)
        lambdas_k = torch.from_numpy(lambdas_k.astype(np.float32))
        self.lambdas_k = lambdas_k
        return lambdas_k

    def forward(self,z):
        v_z = torch.exp(-pairwise_dist2_torch(z, self.centers_rbf) * self.lambdas_k.T)
        res = v_z @ self.W_rbf.T + self.zeta_rbf
        return res
    
def pairwise_dist2_torch(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, 2).sum(2)
    return dist

    

    

        