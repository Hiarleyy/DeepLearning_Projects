import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance_matrix



# Coordenadas fixas das antenas
antenas = np.array([
    [-23.5, -46.6],
    [-23.55, -46.65],
    [-23.52, -46.52]
])

# Gerar posições iniciais aleatórias para os usuários
np.random.seed(42)
latitude = np.random.uniform(-23.6, -23.4, 100)
longitude = np.random.uniform(-46.7, -46.5, 100)
user_locations = np.column_stack((latitude, longitude))
# Gerar posições iniciais aleatórias para os usuários no plano
num_users = 20
user_locations = np.random.rand(num_users, 2) * np.array([0.2, 0.2]) + np.array([-23.6, -46.7])
# Parâmetros de movimentação
step_size = 0.05  # Tamanho do passo (fração da distância para a antena)
min_distance = 0.01  # Distância mínima para a antena

# Preparação para a animação
fig, ax = plt.subplots(figsize=(10, 6))
scat_users = ax.scatter(user_locations[:, 0], user_locations[:, 1], c='blue', label="Usuários")
scat_antennas = ax.scatter(antenas[:, 0], antenas[:, 1], c='red', marker='x', s=100, label="Antenas (fixas)")
ax.set_xlim(-23.6, -23.4)
ax.set_ylim(-46.7, -46.5)
ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.set_title("Movimento dos Usuários em Direção às Antenas")
ax.legend()

# Função de atualização para a animação
min_user_distance = 0.05  # Distância mínima entre usuários
forca_repulsao = 0.01   # Força da repulsão

initial_user_locations = user_locations.copy()
direction_vectors = []

def update(frame):
    global user_locations

    # Calcular distâncias para as antenas
    dist_matrix = distance_matrix(user_locations, antenas)
    nearest_antennas = np.argmin(dist_matrix, axis=1)
    distances_to_nearest = dist_matrix[np.arange(len(user_locations)), nearest_antennas]

    # Inicializar vetor de movimento
    movement_vectors = np.zeros_like(user_locations)

    # Movimento em direção às antenas
    for i, antenna_idx in enumerate(nearest_antennas):
        if distances_to_nearest[i] > min_distance:
            direction_vector = antenas[antenna_idx] - user_locations[i]
            normalized_vector = direction_vector / np.linalg.norm(direction_vector)
            movement_vectors[i] += step_size * normalized_vector * distances_to_nearest[i]

            if frame == 0:
                direction_vectors.append(direction_vector)
        else:
            if frame == 0:
                direction_vectors.append([0, 0])  # Usuário já está próximo da antena

    # Repulsão entre usuários
    for i in range(len(user_locations)):
        for j in range(i + 1, len(user_locations)):
            delta = user_locations[i] - user_locations[j]
            distance = np.linalg.norm(delta)
            if distance < min_user_distance and distance > 0:
                repulsao = (delta / distance) * (min_user_distance - distance) * forca_repulsao
                movement_vectors[i] += repulsao
                movement_vectors[j] -= repulsao

    # Atualizar posições
    user_locations += movement_vectors

    # Atualizar gráfico
    scat_users.set_offsets(user_locations)
    return scat_users,

# Armazenar as posições iniciais

# Criar a animação
ani = FuncAnimation(fig, update, frames=range(50), interval=200, blit=True)
# Mostrar a animação
plt.show()

##df = pd.DataFrame({
##    'Usuario': np.arange(len(initial_user_locations)),
##    'Posicao_Inicial_X': initial_user_locations[:, 0],
##    'Posicao_Inicial_Y': initial_user_locations[:, 1],
##    'Direcao_X':[vec[0] for vec in direction_vectors],
##    'Direcao_Y':[vec[1] for vec in direction_vectors]
##})
##df.to_csv('usuarios_posicoes_e_vetores.csv', index=False)

# Imprimir posições iniciais e finais
print("Posições iniciais dos usuários:")
for idx, pos in enumerate(initial_user_locations):
    print("--------------------------------------")
    print(f"Usuário {idx}: {pos}")

print("======================================")
print("Posições finais dos usuários:")
for idx, pos in enumerate(user_locations):  # Correção feita aqui
    print("--------------------------------------")
    print(f"Usuário {idx}: {pos}")

# ani.save("movimento_usuarios.mp4", writer="ffmpeg")
