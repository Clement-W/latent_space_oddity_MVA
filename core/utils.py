# partially comes from https://github.com/georgiosarvanitidis/geometric_ml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Compute pairwise distances between torch matrices (RETURNS |x-y|^2)
def pairwise_dist2_torch(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, 2).sum(2)
    return dist


def plot_latent_space(latent_means,labels):
    # Set up the plot
    fig = plt.figure(figsize=(8, 6))

    # Check the number of dimensions in latent_means to determine 2D or 3D plot
    if latent_means.shape[1] == 2:
        ax = fig.add_subplot(111)  # 2D scatter plot
        for label in np.unique(labels):
            points = latent_means[labels == label]
            ax.scatter(points[:, 0], points[:, 1], s=20, label=f'Label {label}')
        ax.set_xlabel('Latent Variable 1')
        ax.set_ylabel('Latent Variable 2')
    elif latent_means.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')  # 3D scatter plot
        for label in np.unique(labels):
            points = latent_means[labels == label]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20, label=f'Label {label}')
        ax.set_xlabel('Latent Variable 1')
        ax.set_ylabel('Latent Variable 2')
        ax.set_zlabel('Latent Variable 3')

    ax.set_title('Latent Space of the VAE')
    plt.legend()
    plt.show()


# Plots easily data in 2d or 3d
def my_plot(x, **kwargs):
    if x.shape[1] == 2:
        plt.scatter(x[:, 0], x[:, 1], **kwargs)
        plt.axis('equal')
    if x.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], **kwargs)


# Generate a uniform meshgrid
def my_meshgrid(x1min, x1max, x2min, x2max, N=10):
    X1, X2 = np.meshgrid(np.linspace(x1min, x1max, N), np.linspace(x2min, x2max, N))
    X = np.concatenate((np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)), axis=1)
    return X


# Plots the measure of the Riemannian manifold
def plot_measure(manifold, linspace_x1, linspace_x2, isLog=True, cmapSel=cm.RdBu_r):
    X1, X2 = np.meshgrid(linspace_x1, linspace_x2)
    X = np.concatenate((np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)), axis=1)
    M = manifold.metric_tensor(X.transpose(), nargout=1)

    if manifold.is_diagonal():
        img = np.reshape(np.sqrt(np.prod(M, axis=1)), X1.shape)
    elif not manifold.is_diagonal():
        N = M.shape[0]
        img = np.zeros((N, 1))
        for n in range(N):
            img[n] = np.sqrt(np.linalg.det(np.squeeze(M[n, :, :])))
        img = img.reshape(X1.shape)

    if isLog:
        img = np.log(img + 1e-10)
    else:
        img = img

    plt.imshow(img, interpolation='gaussian', origin='lower',
               extent=(linspace_x1.min(), linspace_x1.max(), linspace_x2.min(), linspace_x2.max()),
               cmap=cmapSel, aspect='equal')
