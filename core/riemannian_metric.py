import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class Riemannian_metric:
    
    def __init__(self,VAERBF_model):
        self.VAERBF = VAERBF_model

    def measure(self,z,var_rbfn=True):
        M_z = self.compute_riemannian_metric(z,var_rbfn)
        return np.sqrt(np.abs(np.linalg.det(M_z))).reshape(-1,1)

    def compute_riemannian_metric(self,z, var_rbfn):
        N, d = z.size()
        z.requires_grad_(True)  # Enable gradient computation for z

        # Will contain the Riemannian metric tensor at each point of the latent space
        M_z_batch = torch.zeros(N, d, d, device=z.device)

        # Forward pass through the model to get mu and sigma
        mu, _ = self.VAERBF.VAE.decode(z)
        if(var_rbfn):
            beta = self.VAERBF.forward_RBF(z)
            sigma = 1/beta.sqrt()
        else:
            _,logvar = self.VAERBF.VAE.decode(z)
            sigma = torch.exp(0.5 * logvar)

        # Compute gradients for mu and sigma with respect to z
        # Iterates over each dimension on the latent space
        for i in range(d):
            # Compute gradients for the i-th component of z
            if z.grad is not None:
                z.grad.data.zero_()

            # setting grad_outputs to a tensor of ones matching the shape of mu[:, i] or 
            # sigma[:, i] indicates that we want to compute the total gradient
            mu_i_grad, = torch.autograd.grad(outputs=mu[:, i], inputs=z,
                                                grad_outputs=torch.ones_like(mu[:, i]),
                                                create_graph=True, retain_graph=True)
            sigma_i_grad, = torch.autograd.grad(outputs=sigma[:, i], inputs=z,
                                                grad_outputs=torch.ones_like(sigma[:, i]),
                                                create_graph=True, retain_graph=True)

            # Constructing the Riemannian metric tensor
            for j in range(d):

                if z.grad is not None:
                    # to prevent gradient accumulation just in case
                    z.grad.data.zero_()

                
                mu_j_grad, = torch.autograd.grad(outputs=mu[:, j], inputs=z,
                                                    grad_outputs=torch.ones_like(mu[:, j]),
                                                    create_graph=True, retain_graph=True)
                sigma_j_grad, = torch.autograd.grad(outputs=sigma[:, j], inputs=z,
                                                    grad_outputs=torch.ones_like(sigma[:, j]),
                                                    create_graph=True, retain_graph=True)

                # The Riemannian metric tensor at (i, j)
                # The diagonal elements of Mz​ are calculated as the sum of the squares of 
                # the gradients of mu and sigma with respect to the same dimension of z 
                # (This gives us the scale of the space in that dimension.)

                # The off-diagonal elements of Mz​ are calculated as the sum of the products
                # of the gradients of mu and sigma with respect to the different dimensions of z
                # (This gives us the correlation between the dimensions of the space.)
                M_z_batch[:, i, j] = torch.sum(mu_i_grad * mu_j_grad, dim=1) + \
                                        torch.sum(sigma_i_grad * sigma_j_grad, dim=1)

        return M_z_batch
    

    import matplotlib.cm as cm


    def plot_MZ(self,linspace_x1, linspace_x2, isLog=True, cmapSel=cm.RdBu_r,vae_rbfn=True):
        X1, X2 = np.meshgrid(linspace_x1, linspace_x2)
        X = np.concatenate((np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)), axis=1)
        M_z = self.compute_riemannian_metric(torch.Tensor(X),var_rbfn=vae_rbfn)
        M_z = M_z.detach().numpy()
        N = M_z.shape[0]
        img_plot = np.zeros((N, 1))
        for n in range(N):
            img_plot[n] = np.sqrt(np.abs(np.linalg.det(np.squeeze(M_z[n, :, :]))))
        img_plot = img_plot.reshape(X1.shape)

        if isLog:
            # to better vizualize small differences in the determinant
            img_plot = np.log(img_plot + 1e-10)

        img_plot = plt.imshow(img_plot, interpolation='gaussian', origin='lower',
                extent=(linspace_x1.min(), linspace_x1.max(), linspace_x2.min(), linspace_x2.max()),
                cmap=cmapSel, aspect='equal')
        
        # Adding a colorbar
        cbar = plt.colorbar(img_plot)
        cbar.set_label('Log of sqrt(det(M))' if isLog else 'Sqrt(det(M))')

        # Labeling the axes
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Heatmap of Riemannian Metric Tensor Determinant in Latent Space')
