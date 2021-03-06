import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.stats import wishart, chi2

import sys
sys.path.append('../lot')

from utils import generate_gmm_data, kmeans
from latentot import LatentOT

n_components, n_dim = 4, 2
# X = generate_gmm_data(n_components, n_dimensions=2).float()
# Y = generate_gmm_data(n_components, n_dimensions=2).float()
source_cores = np.asarray([[1, 1], [2, -1], [-3, -1], [-4, 1]]) * 8
target_cores = source_cores / 2
X, _ = make_blobs(n_samples=400, centers=source_cores)
Y, _ = make_blobs(n_samples=400, centers=target_cores)
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()

eps = 10
lot = LatentOT(X, Y, 4, 4, eps, eps, eps, device=torch.device('cuda'))
px, py, pz, zx, zy = lot.fit(max_iter=1000)
px, py, pz, zx, zy = px.cpu(), py.cpu(), pz.cpu(), zx.cpu(), zy.cpu()
X, Y = X.cpu(), Y.cpu()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(X[:, 0], X[:, 1], c='b', zorder=10)
ax.scatter(Y[:, 0], Y[:, 1], c='r', zorder=10)
ax.scatter(zx[:, 0], zx[:, 1], c='y', marker='^', zorder=20)
ax.scatter(zy[:, 0], zy[:, 1], c='g', marker='^', zorder=20)
ax.axis('equal')
# plot line connect
for i, p in enumerate(px):
    zi = p.argmax()
    ax.plot([X[i, 0], zx[zi, 0]], [X[i, 1], zx[zi, 1]], c='gray', alpha=0.5)

for i, p in enumerate(py.transpose(1, 0)):
    zi = p.argmax()
    ax.plot([Y[i, 0], zy[zi, 0]], [Y[i, 1], zy[zi, 1]], c='gray', alpha=0.5)

for i, p in enumerate(pz):
    zi = p.argmax()
    ax.plot([zx[i, 0], zy[zi, 0]], [zx[i, 1], zy[zi, 1]], c='black', zorder=15)

for i, p in enumerate(pz.transpose(1, 0)):
    zi = p.argmax()
    ax.plot([zy[i, 0], zx[zi, 0]], [zy[i, 1], zx[zi, 1]], c='black', zorder=15)

plt.savefig('demo.png')
plt.show()
