import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

# Cargar el conjunto de datos fashion_MNIST
print("Cargando el conjunto de datos fashion_MNIST...")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("Datos cargados correctamente.")

# Normalización de los datos (valores entre 0 y 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Aplanar imágenes de 28x28 a vectores de 784 dimensiones
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(f"Dimensión de entrada: {x_train.shape[1]}")

# Dimensión de entrada
input_dim = x_train.shape[1]

# Definir el Stacked Autoencoder
encoding_dim = 128  # Primera capa comprimida
hidden_dim = 64  # Segunda capa comprimida

# Definir la arquitectura del Autoencoder
print("Definiendo la arquitectura del Autoencoder...")
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
encoded = Dense(hidden_dim, activation='relu')(encoded)

decoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Crear el modelo Autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
print("Modelo Autoencoder compilado.")

# Entrenar el Autoencoder
print("Entrenando el Autoencoder...")
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
print("Entrenamiento finalizado.")

# Crear el modelo codificador
print("Extrayendo representaciones comprimidas...")
encoder = Model(input_layer, encoded)
encoded_train = encoder.predict(x_train)
encoded_test = encoder.predict(x_test)
print("Representaciones comprimidas obtenidas.")

# Clasificador con la representación comprimida
print("Entrenando clasificador Random Forest con las representaciones comprimidas...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=2, n_jobs=-1)
clf.fit(encoded_train, y_train)
y_pred = clf.predict(encoded_test)
print("Clasificación completada.")

# Evaluar el rendimiento del clasificador
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del clasificador con Stacked Autoencoder: {accuracy:.4f}')

''' LOG
Datos cargados correctamente.
Dimensión de entrada: 784
Definiendo la arquitectura del Autoencoder...
2025-02-25 15:24:25.952253: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Modelo Autoencoder compilado.
Entrenando el Autoencoder...
Epoch 1/10
235/235 ━━━━━━━━━━━━━━━━━━━━ 7s 19ms/step - loss: 0.0900 - val_loss: 0.0310
Epoch 2/10
235/235 ━━━━━━━━━━━━━━━━━━━━ 4s 17ms/step - loss: 0.0272 - val_loss: 0.0185
Epoch 3/10
235/235 ━━━━━━━━━━━━━━━━━━━━ 4s 17ms/step - loss: 0.0178 - val_loss: 0.0144
Epoch 4/10
235/235 ━━━━━━━━━━━━━━━━━━━━ 4s 17ms/step - loss: 0.0142 - val_loss: 0.0122
Epoch 5/10
235/235 ━━━━━━━━━━━━━━━━━━━━ 4s 17ms/step - loss: 0.0124 - val_loss: 0.0110
Epoch 6/10
235/235 ━━━━━━━━━━━━━━━━━━━━ 4s 18ms/step - loss: 0.0111 - val_loss: 0.0099
Epoch 7/10
235/235 ━━━━━━━━━━━━━━━━━━━━ 4s 18ms/step - loss: 0.0099 - val_loss: 0.0091
Epoch 8/10
235/235 ━━━━━━━━━━━━━━━━━━━━ 4s 17ms/step - loss: 0.0091 - val_loss: 0.0085
Epoch 9/10
235/235 ━━━━━━━━━━━━━━━━━━━━ 4s 18ms/step - loss: 0.0085 - val_loss: 0.0079
Epoch 10/10
235/235 ━━━━━━━━━━━━━━━━━━━━ 4s 18ms/step - loss: 0.0081 - val_loss: 0.0075
Entrenamiento finalizado.
Extrayendo representaciones comprimidas...
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 3ms/step   
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step  
Representaciones comprimidas obtenidas.
Entrenando clasificador Random Forest con las representaciones comprimidas...
Clasificación completada.
Accuracy del clasificador con Stacked Autoencoder: 0.9461
'''