#%%
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

#%%
users = {}

for i in range(1, 31):
    users[f'user{i}'] = {
        'ue': i,
        'Posicao_Inicial_X': random.randint(-300, 300),
        'Posicao_Inicial_Y': random.randint(-300, 300)
    }

print(users)
# %%
users
positions = [(user['Posicao_Inicial_X'], user['Posicao_Inicial_Y']) for user in users.values()]
#%%
positions
# %%
antenas = np.array([
    [200, -300],
    [200, 200],
    [-300, 200],
    [-300, -300]
])


# Plot users
x_users, y_users = zip(*positions)
plt.scatter(x_users, y_users, c='blue', label='Users')

# Plot antennas
x_antenas, y_antenas = antenas[:, 0], antenas[:, 1]
plt.scatter(x_antenas, y_antenas, c='red', label='Antennas')

# Add grid
plt.grid(True)

# Add legend
plt.legend()

# Show plot
plt.xlabel('X Pos')
plt.ylabel('Y Pos')
plt.title('Antenas e Usuários')
plt.show()

####### CLUSTERING ########
#%%
kmeans = KMeans(n_clusters=4, random_state=0).fit(positions)
kmeans.cluster_centers_
# %%
# Plot the clustered users
plt.scatter(x_users, y_users, c=kmeans.labels_, cmap='viridis', label='Users')

# Plot the cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='green', s=200, alpha=0.75, label='Cluster Centers')

# Plot antennas
plt.scatter(x_antenas, y_antenas, c='red', label='Antennas')

# Add grid
plt.grid(True)

# Add legend
plt.legend()

# Show plot
plt.xlabel('X Pos')
plt.ylabel('Y Pos')
plt.title('Clusterização de Usuários e Antenas')
plt.show()

# %%
# Create a DataFrame with users and their positions
df_users = pd.DataFrame(users).T
df_users['Cluster'] = kmeans.labels_

df_users
# %%

df_users[['Posicao_Inicial_X', 'Posicao_Inicial_Y']]
antena3 = [-300, -300]

# Calculate distances in x and y from each user to antena3
df_users['Distance_X_to_Antena3'] = df_users['Posicao_Inicial_X'] - antena3[0]
df_users['Distance_Y_to_Antena3'] = df_users['Posicao_Inicial_Y'] - antena3[1]

# Show the distances
df_users[['Posicao_Inicial_X', 'Posicao_Inicial_Y', 'Distance_X_to_Antena3', 'Distance_Y_to_Antena3']]

# %%
