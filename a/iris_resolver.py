#%%
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

#%%
df = pd.read_csv('datasets\iris\Iris.csv')


X= df.iloc[:,1:5].astype('float')

Y= df.iloc[:,5]

encoder = LabelEncoder()
encoded = encoder.fit_transform(Y)

y = to_categorical(encoded)
print(encoded)
print(X)
# %%
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X,y,epochs=2000)

# %%
while True:
    sepal_length = input("Sepal Length:")
    sepal_width = input("Sepal Width:")
    petal_length = input("Petal Length:")
    petal_width = input("Petal Width:")
    
    data = np.array([[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]])
    
    result = model.predict(data)
    
    predicted_class = np.argmax(result)
    
    if predicted_class == 0:
        print("Iris-setosa")
    elif predicted_class == 1:
        print("Iris-versicolor")
    elif predicted_class == 2:
        print("Iris-virginica")
    else:
        print("Unknown class")
# %%
