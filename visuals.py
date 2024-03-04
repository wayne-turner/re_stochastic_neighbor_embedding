import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# t-SNE (t-distributed stochastic neighbor embedding) with Gaussian Mixture

fig = plt.figure(figsize=(12, 8))
fig.patch.set_facecolor('#121212')

ax = fig.add_subplot(111, projection='3d', facecolor='#121212')
cmap = plt.get_cmap('viridis')

norm = plt.Normalize(vmin=np.min(cluster_labels), vmax=np.max(cluster_labels))
colors = cmap(norm(cluster_labels))

for cluster, color in zip(np.unique(cluster_labels), colors):
    indices = cluster_labels == cluster
    ax.scatter(
        features_3d[indices, 0],
        features_3d[indices, 1],
        features_3d[indices, 2],
        s=200,
        c=colors[indices],
        label=cluster_labels_str[cluster],
        edgecolors='w',
        linewidths=0.6
    )

legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.setp(legend.get_texts(), color='w')

ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')

ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.grid(True)

plt.subplots_adjust(right=0.75)

def update(angle):
    ax.view_init(azim=angle)

ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=200)
ani.save('tSNE_rotation.gif', writer='imagemagick', fps=20)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

silhouette_scores = []
n_components_range = range(2, 29, 2)

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0).fit(features)
    cluster_labels = gmm.predict(features)
    score = silhouette_score(features, cluster_labels)
    silhouette_scores.append(score)

# silhouette scores
plt.figure(figsize=(12, 8), facecolor='#121212')
plt.plot(n_components_range, silhouette_scores, marker='o', color=plt.cm.viridis(0.95))
plt.xlabel('Components', color='white')
plt.ylabel('Silhouette', color='white')
plt.xticks(n_components_range, color='white') 
plt.yticks(color='white')
plt.grid(True, color='gray')  
plt.gca().set_facecolor('#121212')

for spine in plt.gca().spines.values():
    spine.set_edgecolor('white')


plt.savefig('silhouette_vs_components.png') 
