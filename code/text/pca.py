import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg

# matplotlib.rcParams.update({'font.size': 16})

mean = (2, 1)
cov = [[1.5, 1], [2, 3.5]]
data = np.random.multivariate_normal(mean, cov, 2500)

x = [s[0] for s in data]
y = [s[1] for s in data]

data_centered = data - mean

cov_matrix = np.cov(data_centered, rowvar=False)
eigvals, eigvecs = linalg.eigh(cov_matrix)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
eigvals = eigvals[idx]

print(eigvals)
print(eigvecs)

data_pca = np.dot(eigvecs.T, data_centered.T).T

px = [s[0] for s in data_pca]
py = [s[1] for s in data_pca]

vx = [v[0] for v in eigvecs.T]
vy = [v[1] for v in eigvecs.T]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
ax.set_aspect(1)
ax.scatter(x, y, c='b')
ax.quiver(mean[0], mean[1], eigvals[0] * vx[0], eigvals[0] * vy[0], color='y', angles='xy', scale_units='xy', scale=1.5,
          width=0.014, headwidth=3, headlength=4, headaxislength=3.5)
ax.quiver(mean[0], mean[1], eigvals[1] * vx[1], eigvals[1] * vy[1], color='y', angles='xy', scale_units='xy', scale=0.5,
          width=0.014, headwidth=3, headlength=4, headaxislength=3.5)
ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_xlim([-8, 8])
ax.set_ylim([-8, 8])
plt.show()

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
ax.set_aspect(1)
ax.scatter(px, py, c='g')
ax.set_xlabel('PC1', fontsize=20)
ax.set_ylabel('PC2', fontsize=20)
ax.set_xlim([-8, 8])
ax.set_ylim([-8, 8])
plt.show()
