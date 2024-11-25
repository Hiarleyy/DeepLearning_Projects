#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Coordenadas fixas das antenas
antenas = np.array([
    [-23.5, -46.6],
    [-23.55, -46.65],
    [-23.52, -46.52],
    [-23.48, -46.58],
    [-23.53, -46.63]
])
#%%
# Gerar posições iniciais aleatórias para os usuários
np.random.seed(1)
latitude = np.random.uniform(-23.6, -23.4, 100)
longitude = np.random.uniform(-46.7, -46.5, 100)
user_locations = np.column_stack((latitude, longitude))

# Parâmetros de movimentação
step_size = 0.05  # Tamanho do passo (fração da distância para a antena)
min_distance = 0.01  # Distância mínima para a antena, que define quando o usuário para de se mover

# Simulação do movimento dos usuários
for _ in range(50):  # Limitar a um máximo de 50 iterações
    # Calcular distância de cada usuário para cada antena
    dist_matrix = distance_matrix(user_locations, antenas)
    
    # Encontrar a antena mais próxima de cada usuário
    nearest_antennas = np.argmin(dist_matrix, axis=1)
    distances_to_nearest = dist_matrix[np.arange(len(user_locations)), nearest_antennas]
    
    # Mover cada usuário em direção à antena mais próxima
    for i, antenna_idx in enumerate(nearest_antennas):
        if distances_to_nearest[i] > min_distance:  # Mover apenas se ainda estiver longe da antena
            direction_vector = antenas[antenna_idx] - user_locations[i]
            normalized_vector = direction_vector / np.linalg.norm(direction_vector)
            user_locations[i] += step_size * normalized_vector * distances_to_nearest[i]
    
    # Visualizar o movimento a cada 10 iterações
    if _ % 10 == 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(user_locations[:, 0], user_locations[:, 1], c=nearest_antennas, cmap='viridis', marker='o', label="Usuários")
        plt.scatter(antenas[:, 0], antenas[:, 1], c='red', marker='x', s=100, label="Antenas (fixas)")
        plt.xlabel("Latitude")
        plt.ylabel("Longitude")
        plt.title(f"Movimento dos Usuários em Direção às Antenas - Iteração {_}")
        plt.legend()
        plt.show()

# %%