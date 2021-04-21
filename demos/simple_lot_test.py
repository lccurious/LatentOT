import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from lot import LOT

n_components, n_dim = 4, 2
# X = generate_gmm_data(n_components, n_dimensions=2).float()
# Y = generate_gmm_data(n_components, n_dimensions=2).float()
source_cores = np.asarray([[1, 1], [2, -1], [-3, -1], [-4, 1]]) * 4
target_cores = source_cores / 2
X, _ = make_blobs(n_samples=400, centers=source_cores)
center_x = X.mean(axis=0)
Y, _ = make_blobs(n_samples=400, centers=target_cores)
center_y = Y.mean(axis=0)

eps = 4
lot_solver = LOT(n_cluster_source=4, n_cluster_target=4,
                 epsilon=eps, epsilon_z=eps, intensity=[1, 1, 1], floyditer=50)

px, py, pz, p, pot, zx, zy, t, converrlist = lot_solver(X, Y)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(X[:, 0], X[:, 1], c='b', zorder=10)
ax.scatter(Y[:, 0], Y[:, 1], c='r', zorder=10)
ax.scatter(zx[:, 0], zx[:, 1], c='y', marker='^', zorder=20)
ax.scatter(zy[:, 0], zy[:, 1], c='g', marker='^', zorder=20)
ax.axis('equal')

# plot line connect
ax.scatter(center_x[0], center_x[1], marker='*', s=200)
ax.scatter(center_y[0], center_y[1], marker='*', s=200)
for i, (pt_x, pt_y) in enumerate(zip(zx, zy)):
    ax.plot([center_x[0], pt_x[0]], [center_x[1], pt_x[1]], c='k')
    ax.plot([center_y[0], pt_y[0]], [center_y[1], pt_y[1]], c='k')

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
