#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix ## Função para o cálculo de distancia relativa

##%%
## Definição das coordenadas Fixas das antenas

antenas = np.array([
    [-23.5, -46.6],
    [-23.55, -46.65],
    [-23.52, -46.52],
    [-23.48, -46.58],
    [-23.53, -46.63]
])

#%%

# Posicionamento aleatório dos usuários
np.random.seed(1)# Seed da simulação 

latitude = np.random.uniform(-23.6, -23.4, 100) # Posicionamento X dos usuarios
longitude = np.random.uniform(-46.7, -46.5, 100) # Posicionamento Y dos usuarios
#Definição do posicionamento aleatório dos usuários
user_locations = np.column_stack((latitude, longitude)) 


step_size = 0.5 # tamanho do passo (fração de distância pra antena)
min_distance = 0.01 #Distancia quando o usuário para de se mover

# Movimento dos usuários

for _ in range(50): # range(50) é o limite de iterações do laço
    dist_matrix = distance_matrix(user_locations, antenas) # Calculo da distancia de cada usuario para cada antena

    ## CALCULO DA ANTENA MAIS PROXIMA DE CADA USUARIO
    nearest_antena = np.argmin(dist_matrix, axis=1) # Encontrar a antena mais próxima de cada 
    distances_to_nearest = dist_matrix[np.arange(len(user_locations)), nearest_antena] # Distancia do usuario para a antena mais proxima


    for i, antena_id in enumerate (nearest_antena):
        if distances_to_nearest[i] > min_distance:
            direction_vector = antenas[antena_id] - user_locations[i]
            normalized_vector = direction_vector / np.linalg.norm(direction_vector)
            user_locations[i] += step_size * normalized_vector * distances_to_nearest[i]

    if _ % 10 == 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(user_locations[:, 0], user_locations[:, 1], c=nearest_antena, cmap='viridis', marker='o', label="Usuários")
        plt.scatter(antenas[:, 0], antenas[:, 1], c='red', marker='x', s=100, label="Antenas (fixas)")
        plt.xlabel("Latitude")
        plt.ylabel("Longitude")
        plt.title(f"Movimento dos Usuários em Direção às Antenas - Iteração {_}")
        plt.legend()
        plt.show()
# %%
