#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform

# Ponto central (antena)
central_point = np.array([0.5, 0.5])

# Gerar dados de exemplo para 10 usuários ao redor do ponto central (posição inicial aleatória)
np.random.seed(42)
num_users = 10
angles = np.random.uniform(0, 2 * np.pi, num_users)
radii = np.random.uniform(0, 0.1, num_users)  # Distância máxima de 0.1 do ponto central

initial_positions = np.zeros((num_users, 2))
initial_positions[:, 0] = central_point[0] + radii * np.cos(angles)
initial_positions[:, 1] = central_point[1] + radii * np.sin(angles)

# Escalar os dados para o intervalo [0, 1]
scaler = MinMaxScaler()
positions_scaled = scaler.fit_transform(initial_positions)

# Definir a arquitetura do autoencoder
input_dim = positions_scaled.shape[1]
encoding_dim = 2  # Dimensão do espaço latente

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = Model(input_layer, decoder)

# Compilar o modelo
autoencoder.compile(optimizer='adam', loss='mse')

# Treinar o autoencoder
autoencoder.fit(positions_scaled, positions_scaled, epochs=500, batch_size=2, shuffle=True, verbose=0)

# Obter a codificação das posições
encoder_model = Model(input_layer, encoder)
encoded_positions = encoder_model.predict(positions_scaled)

# Implementar um algoritmo para otimização das posições baseado nas codificações
def optimize_positions(encoded_positions, central_point, min_distance=0.3, min_distance_to_central=0.3, max_iters=1000):
    optimized_encoded_positions = np.copy(encoded_positions)
    n_points = optimized_encoded_positions.shape[0]

    def calculate_distances(positions):
        return squareform(pdist(positions))

    for _ in range(max_iters):
        distances = calculate_distances(optimized_encoded_positions)
        central_distances = np.linalg.norm(optimized_encoded_positions - central_point, axis=1)
        
        # Verificar se todas as distâncias são maiores ou iguais à distância mínima
        if (np.all(distances >= min_distance) or np.all(distances == 0)) and np.all(central_distances >= min_distance_to_central):
            break
        
        # Ajustar as posições que não atendem ao critério de distância mínima
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if distances[i, j] < min_distance:
                    # Mover o ponto j para longe do ponto i
                    diff = optimized_encoded_positions[j] - optimized_encoded_positions[i]
                    diff_norm = np.linalg.norm(diff)
                    if diff_norm > 0:
                        diff = diff / diff_norm * (min_distance - distances[i, j])
                        optimized_encoded_positions[j] += diff

            # Ajustar a posição se estiver muito perto do ponto central
            if central_distances[i] < min_distance_to_central:
                diff = optimized_encoded_positions[i] - central_point
                diff_norm = np.linalg.norm(diff)
                if diff_norm > 0:
                    diff = diff / diff_norm * (min_distance_to_central - central_distances[i])
                    optimized_encoded_positions[i] += diff

    return optimized_encoded_positions

# Normalizar o ponto central (antena) usando o mesmo scaler
central_point_scaled = scaler.transform(central_point.reshape(1, -1)).flatten()

optimized_encoded_positions = optimize_positions(encoded_positions, central_point_scaled)

# Decodificar as posições otimizadas
decoder_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder_model = Model(decoder_input, decoder_layer(decoder_input))

optimized_positions_scaled = decoder_model.predict(optimized_encoded_positions)

# Reescalar para as coordenadas originais
optimized_positions = scaler.inverse_transform(optimized_positions_scaled)

# Plotar as posições originais e otimizadas em um mapa com grid
plt.figure(figsize=(10, 10))
plt.grid(True)

# Ponto central (antena)
plt.scatter(central_point[0], central_point[1], color='green', s=200, marker='x', label='Ponto Central (Antena)')

# Posições originais
plt.scatter(initial_positions[:, 0], initial_positions[:, 1], color='red', label='Posições Originais')

# Posições otimizadas
plt.scatter(optimized_positions[:, 0], optimized_positions[:, 1], color='blue', label='Posições Otimizadas')

# Adicionar legendas
for i, pos in enumerate(initial_positions):
    plt.text(pos[0], pos[1], f'O{i}', fontsize=9, ha='right', color='red')

for i, pos in enumerate(optimized_positions):
    plt.text(pos[0], pos[1], f'N{i}', fontsize=9, ha='right', color='blue')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Posições de Usuários ao Redor da Antena')
plt.legend()
plt.show()

# %%
