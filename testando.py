#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Leitura do arquivo .txt
df = pd.read_csv('datasets\DlCrtlSinr.txt', sep="\t")
df2 = pd.read_csv('datasets\simulation.csv', sep=",")
# Exibir as primeiras linhas
print(df.head())
#%%
# Definir variáveis de entrada (X) e saída (y)
X = df2[['x_pos', 'y_pos']]
y = df['SINR(dB)']
#%%
# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Criar o modelo Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
# Treinar o modelo
rf.fit(X_train, y_train)

# Fazer previsões no conjunto de testes
y_pred = rf.predict(X_test)
# Calcular o erro médio quadrado
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# %%
