#%% Importação de bibliotecas
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

#%% Gerando usuários com posições iniciais aleatórias
users = {}

for i in range(1, 4):
    users[f'user{i}'] = {
        'ue': i,
        'Posicao_Inicial_X': random.randint(-200, 200),
        'Posicao_Inicial_Y': random.randint(-200, 200)
    }

positions = [(user['Posicao_Inicial_X'], user['Posicao_Inicial_Y']) for user in users.values()]

#%% Coordenadas das antenas
antenas = np.array([
    [100, 100],
    [-100, 100],
])

# Plotando usuários e antenas
x_users, y_users = zip(*positions)
x_antenas, y_antenas = antenas[:, 0], antenas[:, 1]

plt.scatter(x_users, y_users, c='blue', label='Users')
plt.scatter(x_antenas, y_antenas, c='red', label='Antennas')
plt.grid(True)
plt.legend()
plt.xlabel('X Pos')
plt.ylabel('Y Pos')
plt.title('Antenas e Usuários')
plt.show()

#%% Aplicando KMeans para clusterizar os usuários
kmeans = KMeans(n_clusters=2, random_state=0).fit(positions)
centers = kmeans.cluster_centers_

# Plotando usuários clusterizados
plt.scatter(x_users, y_users, c=kmeans.labels_, cmap='viridis', label='Users')
plt.scatter(centers[:, 0], centers[:, 1], c='green', s=200, alpha=0.75, label='Cluster Centers')
plt.scatter(x_antenas, y_antenas, c='red', label='Antennas')
plt.grid(True)
plt.legend()
plt.xlabel('X Pos')
plt.ylabel('Y Pos')
plt.title('Clusterização de Usuários e Antenas')
plt.show()

#%% Criando o DataFrame dos usuários e adicionando os clusters
df_users = pd.DataFrame(users).T
df_users['Cluster'] = kmeans.labels_

#%% Calculando deslocamento em direção aos clusters
df_users['Deslocamento_X'] = 0  # Inicializando coluna para deslocamento no eixo X
df_users['Deslocamento_Y'] = 0  # Inicializando coluna para deslocamento no eixo Y

# Definindo velocidades aleatórias entre 0 e 1
velocidades = np.random.uniform(0, 1, len(df_users))

# Calculando os deslocamentos
for idx, row in df_users.iterrows():
    cluster_idx = row['Cluster']  # Identificar o cluster do usuário
    cluster_x, cluster_y = centers[cluster_idx]  # Coordenadas do cluster do usuário
    
    # Calcular a direção de movimento em relação ao cluster
    direction_x = cluster_x - row['Posicao_Inicial_X']
    direction_y = cluster_y - row['Posicao_Inicial_Y']
    
    # Normalizar direção
    magnitude = np.sqrt(direction_x**2 + direction_y**2)
    normalized_direction_x = direction_x / magnitude
    normalized_direction_y = direction_y / magnitude
    
    # Calcular deslocamento com base na velocidade
    deslocamento_x = normalized_direction_x * velocidades[row['ue'] - 1]
    deslocamento_y = normalized_direction_y * velocidades[row['ue'] - 1]
    
    # Atualizar DataFrame com os deslocamentos
    df_users.at[idx, 'Deslocamento_X'] = deslocamento_x
    df_users.at[idx, 'Deslocamento_Y'] = deslocamento_y

#%% Exibindo o DataFrame com deslocamentos
print(df_users[['Posicao_Inicial_X', 'Posicao_Inicial_Y', 'Cluster', 'Deslocamento_X', 'Deslocamento_Y']])

#%% Calculando as posições finais
df_users['Posicao_Final_X'] = df_users['Posicao_Inicial_X'] + df_users['Deslocamento_X']
df_users['Posicao_Final_Y'] = df_users['Posicao_Inicial_Y'] + df_users['Deslocamento_Y']

# Exibindo as posições finais
print("Posições finais dos usuários após deslocamento:")
print(df_users[['Posicao_Final_X', 'Posicao_Final_Y']])

#%% Plotando usuários após o deslocamento
plt.scatter(df_users['Posicao_Final_X'], df_users['Posicao_Final_Y'], c=df_users['Cluster'], cmap='viridis', label='Moved Users')
plt.scatter(centers[:, 0], centers[:, 1], c='green', s=200, alpha=0.75, label='Cluster Centers')
plt.scatter(x_antenas, y_antenas, c='red', label='Antennas')
plt.grid(True)
plt.legend()
plt.xlabel('X Pos')
plt.ylabel('Y Pos')
plt.title('Movimento dos Usuários em Direção aos Clusters')
plt.show()
