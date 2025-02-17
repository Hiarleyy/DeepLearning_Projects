#%%
import matplotlib.pyplot as plt
from sklearn import neighbors as knn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer ##  importação do dataset de cancer de mama
from sklearn.metrics import accuracy_score
#%%
df = load_breast_cancer() ## carregando o dataset

X = df.data
y = df.target

## Divisao de dados de treino e teste -- (70%/30%)
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.3,random_state=11)
#%%

## Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%% VALORES DE K A SEREM ATRIBUIDOS
Kvalues= [1,3,4,7,9,12,15,20]
acuracia = []
#%% Treinamento utilizando diferentes valores de K

for k in Kvalues:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    #Calculando acuracia do modelo
    acc = accuracy_score(y_test, y_pred)
    acuracia.append(acc)
    print (f"K={k} - Acuracia: {accuracy_score(y_test, y_pred)}")

#%% Plotagem da acurácia vs K
plt.figure(figsize=(12, 6))
plt.plot(Kvalues, acuracia, marker='o', linestyle='-', color='blue', label='Acuracia')
plt.xlabel("Número de Vizinhos (K)")
plt.ylabel("Acurácia")
plt.title("Acurácia com diferentes valores de K")
plt.xticks(Kvalues)
plt.legend()
plt.grid(True)
plt.show()
# %%
