#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
# Definir o ambiente
class Environment:
    def __init__(self, width, height, antenna_position):
        self.width = width
        self.height = height
        self.antenna_position = antenna_position
    
    def calculate_sinr(self, position):
        distance = np.sqrt((position[0] - self.antenna_position[0])**2 + (position[1] - self.antenna_position[1])**2)
        sinr = np.exp(-distance / (0.1 * self.width))  # Função exponencial inversa
        return sinr
    
    def get_reward(self, position):
        sinr = self.calculate_sinr(position)
        return sinr if sinr > 0.7 else 0
    
    def is_terminal_state(self, position):
        return self.calculate_sinr(position) >= 0.7

# Implementar o Q-Learning
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.env = env
        self.lr = learning_rate
        self.df = discount_factor
        self.er = exploration_rate
        self.ed = exploration_decay
        self.q_table = np.zeros((env.width, env.height, 4))
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left
    
    def choose_action(self, state):
        if np.random.random() < self.er:
            return np.random.choice(len(self.actions))
        return np.argmax(self.q_table[state[0], state[1]])
    
    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        td_target = reward + self.df * self.q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.lr * td_error
    
    def train(self, episodes):
        for episode in range(episodes):
            state = (np.random.randint(0, self.env.width), np.random.randint(0, self.env.height))
            while not self.env.is_terminal_state(state):
                action = self.choose_action(state)
                next_state = (state[0] + self.actions[action][0], state[1] + self.actions[action][1])
                next_state = (max(0, min(next_state[0], self.env.width - 1)), max(0, min(next_state[1], self.env.height - 1)))
                reward = self.env.get_reward(next_state)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
            self.er *= self.ed
    
    def get_best_path(self, start_state):
        path = [start_state]
        state = start_state
        while not self.env.is_terminal_state(state):
            action = np.argmax(self.q_table[state[0], state[1]])
            next_state = (state[0] + self.actions[action][0], state[1] + self.actions[action][1])
            next_state = (max(0, min(next_state[0], self.env.width - 1)), max(0, min(next_state[1], self.env.height - 1)))
            path.append(next_state)
            state = next_state
        return path

# Configurações do ambiente
width = 10
height = 10
antenna_position = (width // 2, height // 2)
env = Environment(width, height, antenna_position)

# Treinamento do agente
agent = QLearningAgent(env)
agent.train(1000)

# Trajetória aleatória
def get_random_path(start_state, env):
    path = [start_state]
    state = start_state
    while not env.is_terminal_state(state):
        action = np.random.choice(len(agent.actions))
        next_state = (state[0] + agent.actions[action][0], state[1] + agent.actions[action][1])
        next_state = (max(0, min(next_state[0], env.width - 1)), max(0, min(next_state[1], env.height - 1)))
        path.append(next_state)
        state = next_state
    return path

start_state = (0, 0)
random_path = get_random_path(start_state, env)
best_path = agent.get_best_path(start_state)

# Plotar as trajetórias
plt.figure(figsize=(10, 10))
plt.plot(*zip(*random_path), marker='o', color='r', label='Trajetória Aleatória')
plt.plot(*zip(*best_path), marker='o', color='b', label='Trajetória ML')
plt.scatter(*antenna_position, color='g', s=200, label='Antena')
plt.xlim(-1, width)
plt.ylim(-1, height)
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajetórias de Movimentação do Usuário')
plt.grid()
plt.show()

# %%
