#%%
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
#%%
x = np.array([[0.1],[0.2],[0.3]])
y = np.array([[0.2],[0.4],[0.6]])

model = Sequential()

model.add(Dense(3, input_dim=1))
model.add(Dense(1))
model.compile(optimizer='sgd', loss='mean_squared_error',metrics=['accuracy'])

model.fit(x,y,epochs=4000)

while True:
    i = input("Digite um numero:")

    t = float(i)

    t = np.asmatrix(t)

    result = model.predict(t)

    print('previsto =>',result[0],'como nota')

