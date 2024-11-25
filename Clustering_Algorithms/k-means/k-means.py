import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance_matrix

# Coordenadas fixas das antenas
antenas = np.array([
    [50, -50],
    [-50, 50]
])

# Gerar posições iniciais aleatórias para os usuários
np.random.seed(42)
num_users = 20
user_locations = np.random.rand(num_users, 2) * np.array([200, 200]) + np.array([-100, -100])

# Armazenar as posições iniciais
initial_user_locations = user_locations.copy()

# Lista para armazenar os vetores direção iniciais
initial_direction_vectors = [None] * num_users  # Garante o mesmo tamanho que num_users

# Parâmetros de movimentação
step_size = 0.05  # Tamanho do passo
min_distance = 15  # Distância mínima para a antena

# Parâmetros para repulsão entre usuários
min_user_distance = 7  # Distância mínima entre usuários
forca_repulsao = 0.05  # Força da repulsão

# Preparação para a animação
fig, ax = plt.subplots(figsize=(10, 6))
scat_users = ax.scatter(user_locations[:, 0], user_locations[:, 1], c='blue', label="Usuários")
scat_antennas = ax.scatter(antenas[:, 0], antenas[:, 1], c='red', marker='x', s=100, label="Antenas (fixas)")

ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.set_title("Movimento dos Usuários em Direção às Antenas")
ax.legend()

def update(frame):
    global user_locations, initial_direction_vectors

    # Calcular distâncias para as antenas
    dist_matrix = distance_matrix(user_locations, antenas)
    nearest_antennas = np.argmin(dist_matrix, axis=1)
    distances_to_nearest = dist_matrix[np.arange(len(user_locations)), nearest_antennas]

    # Inicializar vetor de movimento
    movement_vectors = np.zeros_like(user_locations)

    # Movimento em direção às antenas
    for i, antenna_idx in enumerate(nearest_antennas):
        direction_vector = antenas[antenna_idx] - user_locations[i]
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            normalized_vector = direction_vector / norm
            if distances_to_nearest[i] > min_distance:
                movement_vectors[i] += step_size * normalized_vector * distances_to_nearest[i]
        else:
            normalized_vector = np.array([0, 0])

        # Armazenar o vetor direção na primeira iteração
        if frame == 0 and initial_direction_vectors[i] is None:
            initial_direction_vectors[i] = direction_vector.copy()

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

# Criar a animação
ani = FuncAnimation(fig, update, frames=range(50), interval=200, blit=True)
ani.save("movimento_usuarios.gif", writer="imagemagick")
plt.show()

# Garantir que todos os vetores foram calculados
initial_direction_vectors = np.array(initial_direction_vectors)

# Criar o DataFrame com as informações desejadas
df = pd.DataFrame({
    'Usuario': np.arange(len(initial_user_locations)),
    'Posicao_Inicial_X': initial_user_locations[:, 0].round(5),
    'Posicao_Inicial_Y': initial_user_locations[:, 1].round(5),
    'Direcao_X': initial_direction_vectors[:, 0],
    'Direcao_Y': initial_direction_vectors[:, 1],
})

# Salvar em um arquivo CSV
df.to_csv('Posicoes.csv', index=False)

# Opcional: imprimir o DataFrame
print(df)
