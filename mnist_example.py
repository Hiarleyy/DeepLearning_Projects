#%%
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt 
from tensorflow.keras.utils import to_categorical
#%%
#carrega o Mnist Dataset
(X_train, Y_train),(X_test,Y_test) = mnist.load_data()

plt.imshow(X_train[4], cmap='gray')

#normaliza os dados	
num_pixels = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float')

X_train = X_train / 255
X_test = X_test / 255
#%%
# Convert labels to categorical one-hot encoding
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

classes = y_test.shape[1]

model = Sequential()

model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=100)
# %%
