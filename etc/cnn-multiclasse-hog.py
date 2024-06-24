# Passo 1: Importar as bibliotecas necessárias
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from skimage.feature import hog
from skimage import color, exposure

# Passo 2: Carregar e preparar o conjunto de dados MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar os dados para o intervalo 0-1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Redimensionar os dados para o formato que a CNN espera (batch_size, height, width, channels)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Converter os rótulos em categorias one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Passo 3: Definir a arquitetura da CNN
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='sigmoid')
])

# Passo 4: Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Passo 5: Treinar o modelo
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Passo 6: Avaliar o modelo
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Passo 7: Preparar os dados usando o descritor HOG
def extract_hog_features(images):
    hog_features = []
    for image in images:
        image = np.squeeze(image)  # Remover o canal se houver
        fd = hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=False )
        hog_features.append(fd)
    return np.array(hog_features)

# Extrair características HOG para treinamento e teste
x_train_hog = extract_hog_features(x_train)
x_test_hog = extract_hog_features(x_test)

# Passo 8: Redefinir o modelo para aceitar as características HOG como entrada
model_hog = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar o modelo
model_hog.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo com características HOG
model_hog.fit(x_train_hog, y_train, validation_data=(x_test_hog, y_test), epochs=10)

# Avaliar o modelo com características HOG
loss_hog, accuracy_hog = model_hog.evaluate(x_test_hog, y_test)
print(f'HOG Loss: {loss_hog}, HOG Accuracy: {accuracy_hog}')