# Importar bibliotecas necessárias
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from skimage.feature import hog
from skimage import exposure

# Passo 1: Preparar o conjunto de dados MNIST para classificação binária
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Filtrar para manter apenas as classes 0 e 1
filter_indices = np.where((y_train == 0) | (y_train == 1))
x_train, y_train = x_train[filter_indices], y_train[filter_indices]
filter_indices = np.where((y_test == 0) | (y_test == 1))
x_test, y_test = x_test[filter_indices], y_test[filter_indices]

# Normalizar os dados
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Converter os rótulos para categorias binárias
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Passo 2: Definir a arquitetura da CNN
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Usar softmax para classificação binária
])

# Passo 3: Compilar e treinar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Passo 4: Preparar os dados usando o descritor HOG
def extract_hog_features(images):
    hog_features = []
    for image in images:
        image = np.squeeze(image)
        fd = hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=False)
        hog_features.append(fd)
    return np.array(hog_features)

x_train_hog = extract_hog_features(x_train)
x_test_hog = extract_hog_features(x_test)

# Passo 5: Treinar o modelo com características HOG
model_hog = Sequential([
    Dense(128, activation='relu', input_shape=(x_train_hog.shape[1],)),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model_hog.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_hog.fit(x_train_hog, y_train, validation_data=(x_test_hog, y_test), epochs=100)