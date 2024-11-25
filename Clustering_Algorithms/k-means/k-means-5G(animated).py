import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance_matrix

# Coordenadas fixas das antenas
antenas = np.array([
    [100, -100],
    [100, 100]
])

# Gerar posições iniciais aleatórias para os usuários
np.random.seed(42)
num_users = 20
# Gerar posições iniciais aleatórias para os usuários
user_locations = np.random.rand(num_users, 2) * np.array([200, 200]) + np.array([-100, -100])

# Armazenar as posições iniciais
initial_user_locations = user_locations.copy()

# Lista para armazenar os vetores direção
direction_vectors = []

# Parâmetros de movimentação
step_size = 0.05  # Tamanho do passo
min_distance = 0.01  # Distância mínima para a antena

# Parâmetros para repulsão entre usuários
min_user_distance = 0.05  # Distância mínima entre usuários
forca_repulsao = 0.01     # Força da repulsão

# Preparação para a animação
fig, ax = plt.subplots(figsize=(10, 6))
scat_users = ax.scatter(user_locations[:, 0], user_locations[:, 1], c='blue', label="Usuários")
scat_antennas = ax.scatter(antenas[:, 0], antenas[:, 1], c='red', marker='x', s=100, label="Antenas (fixas)")

ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.set_title("Movimento dos Usuários em Direção às Antenas")
ax.legend()

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
        direction_vector = antenas[antenna_idx] - user_locations[i]
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            normalized_vector = direction_vector / norm
            if distances_to_nearest[i] > min_distance:
                movement_vectors[i] += step_size * normalized_vector * distances_to_nearest[i]
        else:
            normalized_vector = np.array([0, 0])

        # Armazenar o vetor direção na primeira iteração
        if frame == 0:
            direction_vectors.append(direction_vector.copy())

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
plt.show()

# Criar o DataFrame com as informações desejadas
df = pd.DataFrame({
    'Usuario': np.arange(len(initial_user_locations)),
    'Posicao_Inicial_X': initial_user_locations[:, 0].round(5),
    'Posicao_Inicial_Y': initial_user_locations[:, 1].round(5),

})

df2 = pd.DataFrame({
    'Direcao_X': [vec[0] for vec in direction_vectors],
    'Direcao_Y': [vec[1] for vec in direction_vectors]
})

# Salvar em um arquivo CSV
df.to_csv('Posicoes.csv', index=False)
df2.to_csv('Vetores.csv', index=False)

# Opcional: imprimir o DataFrame
print(df)