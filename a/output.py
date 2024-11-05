#%%
import pandas as pd
import numpy as np

total_records = 9280
total_users = 3
simulation_time = 60  # tempo em segundos
time_step = simulation_time / (total_records // total_users)  # calculando o intervalo de tempo entre registros
#%%
def simulate_user_movement(user_id, initial_pos, velocity, total_steps, time_step):
    times = np.arange(0, total_steps * time_step, time_step)
    x_positions = initial_pos[0] + velocity[0] * times
    y_positions = initial_pos[1] + velocity[1] * times
    user_data = {
        'user_id': [user_id] * total_steps,
        'time': times,
        'x_pos': x_positions,
        'y_pos': y_positions
    }
    return pd.DataFrame(user_data)
#%%
dataframes = []
for user_id in range(1, total_users + 1):
    # Posição inicial aleatória entre 0 e 500
    initial_pos = np.random.uniform(0, 500, 2)
    # Velocidade aleatória entre -2 e 2 (float até 2 casas decimais)
    velocity = np.round(np.random.uniform(-2, 2, 2), 2)
    # Quantidade de registros por usuário
    steps_per_user = total_records // total_users

    user_df = simulate_user_movement(user_id, initial_pos, velocity, steps_per_user, time_step)
    dataframes.append(user_df)

df_simulation = pd.concat(dataframes, ignore_index=True)
print(df_simulation)
# %%
usuario_1 = df_simulation[df_simulation['user_id'] == 1]
usuario_1 
# %%
