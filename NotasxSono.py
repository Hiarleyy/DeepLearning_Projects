import numpy as np
from keras.models import Sequential
from keras.layers import Dense


x  = np.array([[0.8,3.0],[10.0,4.0],[10.0,5.0],[7.0,4.0],[5.0,7.0]])
y = np.array([[7.0],[8.0],[9.5],[4.5],[5.0]])

model = Sequential()

model.add(Dense(3, input_dim=2))
model.add(Dense(1))
model.compile(optimizer='sgd', loss='mean_squared_error',metrics=['accuracy'])
model.fit(x,y,epochs=4000)



while True:
    dormiu = input("Dormiu:")
    estudou = input("Estudou:")
    lista = [float(dormiu),float(estudou)]
    t = np.asmatrix(lista)
    result = model.predict(t)
    print("predicao:",result[0][0])