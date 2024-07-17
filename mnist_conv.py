#%%
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, MaxPooling2D, Flatten
from keras.layers import Conv2D
from keras.datasets import mnist
import matplotlib.pyplot as plt
#%%
#carrega o Mnist Dataset
(X_train, Y_train),(X_test,Y_test) = mnist.load_data()

plt.imshow(X_train[2], cmap='gray')
# %%
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_train = X_train / 255

Y_train = to_categorical(Y_train)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Flatten())
model.add(Dense(Y_train.shape[1], activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=2, batch_size=32)
#%%


