#%%
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from matplotlib.animation import FuncAnimation

#%%
data = []

for i in range(1, 31):
    data.append({
        'ue': i,
        'Posicao_Inicial_X': random.randint(-300, 300),
        'Posicao_Inicial_Y': random.randint(-300, 300)
    })

df = pd.DataFrame(data)
df.reset_index(drop=True, inplace=True)
print(df)
# Create a new column 'prioridade' with 50% 'h', 30% 'm', and 20% 'l'
priorities = ['l'] * 15 + ['m'] * 9 + ['h'] * 6
random.shuffle(priorities)
df['prioridade'] = priorities

print(df)
# %%
df
# %%
colors = {'l': 'green', 'm': 'blue', 'h': 'red'}
plt.figure(figsize=(10, 10))
plt.grid(True)
plt.xlim(-300, 300)
plt.ylim(-300, 300)

for priority in df['prioridade'].unique():
    subset = df[df['prioridade'] == priority]
    plt.scatter(subset['Posicao_Inicial_X'], subset['Posicao_Inicial_Y'], 
                c=colors[priority], label=priority, s=100)

plt.xlabel('Posicao_Inicial_X')
plt.ylabel('Posicao_Inicial_Y')
plt.legend(title='Prioridade')
plt.title('Usuários separados por prioridade')
plt.show()
# %% CLUSTERING
# Separate users by 'prioridade' and shuffle
# Separate users by 'prioridade' and shuffle
df_l = df[df['prioridade'] == 'l'].sample(frac=1).reset_index(drop=True)
df_m = df[df['prioridade'] == 'm'].sample(frac=1).reset_index(drop=True)
df_h = df[df['prioridade'] == 'h'].sample(frac=1).reset_index(drop=True)

# Combine all users and shuffle
all_users = pd.concat([df_l, df_m, df_h], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Ensure each cluster has the same number of users
num_clusters = 3
users_per_cluster = len(all_users) // num_clusters

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(all_users[['Posicao_Inicial_X', 'Posicao_Inicial_Y']])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
all_users['cluster'] = kmeans.fit_predict(scaled_features)

# Adjust clusters to have the same number of users
for cluster in range(num_clusters):
    while len(all_users[all_users['cluster'] == cluster]) > users_per_cluster:
        excess_user = all_users[all_users['cluster'] == cluster].sample(n=1)
        other_clusters = [c for c in range(num_clusters) if c != cluster]
        new_cluster = random.choice(other_clusters)
        all_users.at[excess_user.index[0], 'cluster'] = new_cluster

# Visualize clusters
plt.figure(figsize=(10, 10))
plt.grid(True)
plt.xlim(-300, 300)
plt.ylim(-300, 300)

# Plot cluster regions
x_min, x_max = scaled_features[:, 0].min() - 1, scaled_features[:, 0].max() + 1
y_min, y_max = scaled_features[:, 1].min() - 1, scaled_features[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

# Plot users with different colors for 'prioridade'
colors = {'l': 'green', 'm': 'blue', 'h': 'red'}
for priority in df['prioridade'].unique():
    subset = all_users[all_users['prioridade'] == priority]
    plt.scatter(subset['Posicao_Inicial_X'], subset['Posicao_Inicial_Y'], 
                c=colors[priority], label=priority, s=100)

# Draw different shapes around clusters
shapes = ['o', 's', 'D']  # Circle, square, diamond
for i in range(3):
    subset = all_users[all_users['cluster'] == i]
    plt.scatter(subset['Posicao_Inicial_X'], subset['Posicao_Inicial_Y'],
                edgecolor='black', facecolor='none', s=200, linewidth=1.5, marker=shapes[i], label=f'Cluster {i + 1}')

plt.xlabel('Posicao_Inicial_X')
plt.ylabel('Posicao_Inicial_Y')
plt.legend(title='Prioridade e Clusters')
plt.title('Usuários separados por Prioridade e Cluster')
plt.show()
# %%
# Calculate actual counts of 'prioridade' per cluster
priority_cluster_counts = all_users.groupby(['cluster', 'prioridade']).size().unstack(fill_value=0)
print(priority_cluster_counts)

# Update the original dataframe with cluster information
df['cluster'] = all_users['cluster']
# %%
# Verify if each cluster has 50% 'l', 30% 'm', and 20% 'h'
def check_priority_distribution(cluster_data):
    total_users = len(cluster_data)
    l_count = len(cluster_data[cluster_data['prioridade'] == 'l'])
    m_count = len(cluster_data[cluster_data['prioridade'] == 'm'])
    h_count = len(cluster_data[cluster_data['prioridade'] == 'h'])
    
    l_percentage = l_count / total_users
    m_percentage = m_count / total_users
    h_percentage = h_count / total_users
    
    return l_percentage, m_percentage, h_percentage

for cluster in range(3):
    cluster_data = all_users[all_users['cluster'] == cluster]
    l_percentage, m_percentage, h_percentage = check_priority_distribution(cluster_data)
    print(f"Cluster {cluster + 1}:")
    print(f"  'l' percentage: {l_percentage:.2f}")
    print(f"  'm' percentage: {m_percentage:.2f}")
    print(f"  'h' percentage: {h_percentage:.2f}")
    print()

# %%
def reassign_users_to_balance_clusters(all_users, target_ratios):
    clusters = all_users['cluster'].unique()
    total_users = len(all_users)
    users_per_cluster = total_users // len(clusters)
    
    for cluster in clusters:
        cluster_data = all_users[all_users['cluster'] == cluster]
        while not check_cluster_balance(cluster_data, target_ratios):
            for priority, target_ratio in target_ratios.items():
                current_count = len(cluster_data[cluster_data['prioridade'] == priority])
                target_count = int(users_per_cluster * target_ratio)
                
                if current_count > target_count:
                    excess_users = cluster_data[cluster_data['prioridade'] == priority].sample(n=current_count - target_count)
                    for idx in excess_users.index:
                        other_clusters = [c for c in clusters if c != cluster]
                        new_cluster = random.choice(other_clusters)
                        all_users.at[idx, 'cluster'] = new_cluster
                        cluster_data = all_users[all_users['cluster'] == cluster]
                
                elif current_count < target_count:
                    for donor_cluster in clusters:
                        if donor_cluster != cluster:
                            donor_data = all_users[all_users['cluster'] == donor_cluster]
                            donor_users = donor_data[donor_data['prioridade'] == priority]
                            if len(donor_users) > int(users_per_cluster * target_ratios[priority]):
                                user_to_move = donor_users.sample(n=1)
                                all_users.at[user_to_move.index[0], 'cluster'] = cluster
                                cluster_data = all_users[all_users['cluster'] == cluster]
    
    return all_users

def check_cluster_balance(cluster_data, target_ratios):
    total_users = len(cluster_data)
    for priority, target_ratio in target_ratios.items():
        current_count = len(cluster_data[cluster_data['prioridade'] == priority])
        target_count = int(total_users * target_ratio)
        if current_count != target_count:
            return False
    return True

#%%
target_ratios = {'l': 0.50, 'm': 0.30, 'h': 0.20}
original_users = all_users.copy()
balanced_users = reassign_users_to_balance_clusters(all_users, target_ratios)
#%%
# Verify the new distribution
for cluster in range(3):
    cluster_data = balanced_users[balanced_users['cluster'] == cluster]
    l_percentage, m_percentage, h_percentage = check_priority_distribution(cluster_data)
    print(f"Cluster {cluster + 1} after reassignment:")
    print(f"  'l' percentage: {l_percentage:.2f}")
    print(f"  'm' percentage: {m_percentage:.2f}")
    print(f"  'h' percentage: {h_percentage:.2f}")
    print()
# %%
# Visualize clusters after reassignment
plt.figure(figsize=(10, 10))
plt.grid(True)
plt.xlim(-300, 300)
plt.ylim(-300, 300)

# Plot cluster regions
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

# Plot users with different colors for 'prioridade'
for priority in df['prioridade'].unique():
    subset = balanced_users[balanced_users['prioridade'] == priority]
    plt.scatter(subset['Posicao_Inicial_X'], subset['Posicao_Inicial_Y'], 
                c=colors[priority], label=priority, s=100)

# Draw different shapes around clusters
for i in range(3):
    subset = balanced_users[balanced_users['cluster'] == i]
    plt.scatter(subset['Posicao_Inicial_X'], subset['Posicao_Inicial_Y'],
                edgecolor='black', facecolor='none', s=200, linewidth=1.5, marker=shapes[i], label=f'Cluster {i + 1}')

plt.xlabel('Posicao_Inicial_X')
plt.ylabel('Posicao_Inicial_Y')
plt.legend(title='Prioridade e Clusters')
plt.title('Usuários separados por Prioridade e Cluster após Reatribuição')
plt.show()
# %%
# Visualize changes by highlighting reassigned users
plt.figure(figsize=(10, 10))
plt.grid(True)
plt.xlim(-300, 300)
plt.ylim(-300, 300)

# Plot cluster regions
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

# Plot users with different colors for 'prioridade'
for priority in df['prioridade'].unique():
    subset = balanced_users[balanced_users['prioridade'] == priority]
    plt.scatter(subset['Posicao_Inicial_X'], subset['Posicao_Inicial_Y'], 
                c=colors[priority], label=priority, s=100)

# Highlight reassigned users
reassigned_users = balanced_users[balanced_users['cluster'] != original_users['cluster']]
plt.scatter(reassigned_users['Posicao_Inicial_X'], reassigned_users['Posicao_Inicial_Y'],
            edgecolor='yellow', facecolor='none', s=300, linewidth=2, label='Reassigned Users')

# Draw different shapes around clusters
for i in range(3):
    subset = balanced_users[balanced_users['cluster'] == i]
    plt.scatter(subset['Posicao_Inicial_X'], subset['Posicao_Inicial_Y'],
                edgecolor='black', facecolor='none', s=200, linewidth=1.5, marker=shapes[i], label=f'Cluster {i + 1}')

plt.xlabel('Posicao_Inicial_X')
plt.ylabel('Posicao_Inicial_Y')
plt.legend(title='Prioridade e Clusters')
plt.title('Usuários separados por Prioridade e Cluster após Reatribuição (Reassigned Users Highlighted)')
plt.show()
# %%
# Visualize centroids
plt.figure(figsize=(10, 10))
plt.grid(True)
plt.xlim(-300, 300)
plt.ylim(-300, 300)

# Plot cluster regions
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

# Plot users with different colors for 'prioridade'
for priority in df['prioridade'].unique():
    subset = balanced_users[balanced_users['prioridade'] == priority]
    plt.scatter(subset['Posicao_Inicial_X'], subset['Posicao_Inicial_Y'], 
                c=colors[priority], label=priority, s=100)

# Draw different shapes around clusters
for i in range(3):
    subset = balanced_users[balanced_users['cluster'] == i]
    plt.scatter(subset['Posicao_Inicial_X'], subset['Posicao_Inicial_Y'],
                edgecolor='black', facecolor='none', s=200, linewidth=1.5, marker=shapes[i], label=f'Cluster {i + 1}')

# Plot centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='yellow', s=300, marker='X', edgecolor='black', linewidth=2, label='Centroids')

plt.xlabel('Posicao_Inicial_X')
plt.ylabel('Posicao_Inicial_Y')
plt.legend(title='Prioridade e Clusters')
plt.title('Usuários separados por Prioridade e Cluster com Centróides')
plt.show()






# %%
# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
ax.grid(True)
ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)

# Plot cluster regions
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

# Plot initial positions
scatters = {}
for priority in df['prioridade'].unique():
    subset = balanced_users[balanced_users['prioridade'] == priority]
    scatters[priority] = ax.scatter(subset['Posicao_Inicial_X'], subset['Posicao_Inicial_Y'], 
                                    c=colors[priority], label=priority, s=100)

# Highlight reassigned users
reassigned_users = balanced_users[balanced_users['cluster'] != original_users['cluster']]
reassigned_scat = ax.scatter(reassigned_users['Posicao_Inicial_X'], reassigned_users['Posicao_Inicial_Y'],
                             edgecolor='yellow', facecolor='none', s=300, linewidth=2, label='Reassigned Users')

# Draw different shapes around clusters
for i in range(3):
    subset = balanced_users[balanced_users['cluster'] == i]
    ax.scatter(subset['Posicao_Inicial_X'], subset['Posicao_Inicial_Y'],
               edgecolor='black', facecolor='none', s=200, linewidth=1.5, marker=shapes[i], label=f'Cluster {i + 1}')

# Plot centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centroids[:, 0], centroids[:, 1], c='yellow', s=300, marker='X', edgecolor='black', linewidth=2, label='Centroids')

ax.set_xlabel('Posicao_Inicial_X')
ax.set_ylabel('Posicao_Inicial_Y')
ax.legend(title='Prioridade e Clusters')
ax.set_title('Usuários se movendo para novos Clusters')

# Function to update the scatter plot
def update(frame):
    for priority in df['prioridade'].unique():
        subset = balanced_users[balanced_users['prioridade'] == priority]
        new_positions = subset[['Posicao_Inicial_X', 'Posicao_Inicial_Y']].values * (frame / 100) + \
                        original_users[original_users['prioridade'] == priority][['Posicao_Inicial_X', 'Posicao_Inicial_Y']].values * (1 - frame / 100)
        scatters[priority].set_offsets(new_positions)
    
    new_positions_reassigned = reassigned_users[['Posicao_Inicial_X', 'Posicao_Inicial_Y']].values * (frame / 100) + \
                               original_users.loc[reassigned_users.index][['Posicao_Inicial_X', 'Posicao_Inicial_Y']].values * (1 - frame / 100)
    reassigned_scat.set_offsets(new_positions_reassigned)
    return list(scatters.values()) + [reassigned_scat]

# Create animation
ani = FuncAnimation(fig, update, frames=range(101), interval=100, blit=True)

# Save animation
ani.save('user_reassignment_animation.mp4', writer='ffmpeg')

plt.show()
# %%
# Update the dataframe with new positions and clusters
df.update(balanced_users[['Posicao_Inicial_X', 'Posicao_Inicial_Y', 'cluster']])

# Function to move users towards their cluster centroids
def move_towards_centroids(df, centroids, steps=100):
    for step in range(steps):
        for i, row in df.iterrows():
            cluster = row['cluster']
            centroid = centroids[cluster]
            df.at[i, 'Posicao_Inicial_X'] += (centroid[0] - row['Posicao_Inicial_X']) / steps
            df.at[i, 'Posicao_Inicial_Y'] += (centroid[1] - row['Posicao_Inicial_Y']) / steps
    return df

# Move reassigned users towards their new cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
df = move_towards_centroids(df, centroids)

# Create a figure and axis for the animation
fig, ax = plt.subplots(figsize=(10, 10))
ax.grid(True)
ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)

# Plot cluster regions
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

# Plot initial positions
scatters = {}
for priority in df['prioridade'].unique():
    subset = df[df['prioridade'] == priority]
    scatters[priority] = ax.scatter(subset['Posicao_Inicial_X'], subset['Posicao_Inicial_Y'], 
                                    c=colors[priority], label=priority, s=100)

# Highlight reassigned users
reassigned_users = df[df['cluster'] != original_users['cluster']]
reassigned_scat = ax.scatter(reassigned_users['Posicao_Inicial_X'], reassigned_users['Posicao_Inicial_Y'],
                             edgecolor='yellow', facecolor='none', s=300, linewidth=2, label='Reassigned Users')

# Draw different shapes around clusters
for i in range(3):
    subset = df[df['cluster'] == i]
    ax.scatter(subset['Posicao_Inicial_X'], subset['Posicao_Inicial_Y'],
               edgecolor='black', facecolor='none', s=200, linewidth=1.5, marker=shapes[i], label=f'Cluster {i + 1}')

# Plot centroids
ax.scatter(centroids[:, 0], centroids[:, 1], c='yellow', s=300, marker='X', edgecolor='black', linewidth=2, label='Centroids')

ax.set_xlabel('Posicao_Inicial_X')
ax.set_ylabel('Posicao_Inicial_Y')
ax.legend(title='Prioridade e Clusters')
ax.set_title('Usuários se movendo para novos Clusters')

# Function to update the scatter plot
def update(frame):
    for priority in df['prioridade'].unique():
        subset = df[df['prioridade'] == priority]
        new_positions = subset[['Posicao_Inicial_X', 'Posicao_Inicial_Y']].values * (frame / 100) + \
                        original_users[original_users['prioridade'] == priority][['Posicao_Inicial_X', 'Posicao_Inicial_Y']].values * (1 - frame / 100)
        scatters[priority].set_offsets(new_positions)
    
    new_positions_reassigned = reassigned_users[['Posicao_Inicial_X', 'Posicao_Inicial_Y']].values * (frame / 100) + \
                               original_users.loc[reassigned_users.index][['Posicao_Inicial_X', 'Posicao_Inicial_Y']].values * (1 - frame / 100)
    reassigned_scat.set_offsets(new_positions_reassigned)
    return list(scatters.values()) + [reassigned_scat]

# Create animation
ani = FuncAnimation(fig, update, frames=range(101), interval=100, blit=True)

# Save animation
ani.save('user_reassignment_animation.mp4', writer='ffmpeg')

plt.show()

#%%
# Create a DataFrame to store the displacement information
displacement_data = []

for i, row in df.iterrows():
    original_position = original_users.loc[i, ['Posicao_Inicial_X', 'Posicao_Inicial_Y']].values
    new_position = row[['Posicao_Inicial_X', 'Posicao_Inicial_Y']].values
    displacement_data.append({
        'ue': row['ue'],
        'original_x': original_position[0],
        'original_y': original_position[1],
        'new_x': new_position[0],
        'new_y': new_position[1],
        'vetor_direcao_x': new_position[0] - original_position[0],
        'vetor_direcao_y': new_position[1] - original_position[1]
    })

displacement_df = pd.DataFrame(displacement_data)
print(displacement_df)

# Create a DataFrame to store the centroid information
centroid_data = []

for i, centroid in enumerate(centroids):
    centroid_data.append({
        'cluster': i,
        'centroid_x': centroid[0],
        'centroid_y': centroid[1]
    })

centroid_df = pd.DataFrame(centroid_data)
print(centroid_df)
# %%
displacement_df
# %%
centroid_df
# %%
# Calculate the Euclidean distance for each displacement
displacement_df['modulo_vetor'] = np.sqrt(displacement_df['vetor_direcao_x']**2 + displacement_df['vetor_direcao_y']**2)
print(displacement_df)
# %%
displacement_df
# %%
