import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from skimage.feature import hog
from skimage import exposure

import numpy as np
import matplotlib.pyplot as plt

# Carregar e pré-processar os dados
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Pré-processar os dados
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Função para extrair HOG
def extract_hog_features(images):
    hog_features = []
    for image in images:
        fd, hog_image = hog(image[:, :, 0], orientations=8, pixels_per_cell=(4, 4),
                            cells_per_block=(1, 1), visualize=True)
        hog_features.append(fd)
    hog_features = np.array(hog_features)
    return hog_features

# Visualizando a imagem original e a imagem HOG
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

#ax1.axis('off')
#ax1.imshow(image, cmap=plt.cm.gray)
#ax1.set_title('Imagem Original')
#
## Ajuste de contraste para melhor visualização
#hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
#ax2.axis('off')
#ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
#ax2.set_title('Histograma de Gradientes Orientados')
#plt.show()

# Extrair características HOG dos dados
train_hog_features = extract_hog_features(train_images)
test_hog_features = extract_hog_features(test_images)

# Redimensionar os dados extraídos para o modelo
train_hog_features = np.expand_dims(train_hog_features, axis=-1)
test_hog_features = np.expand_dims(test_hog_features, axis=-1)

# Construir o modelo
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10 classes

# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']) #Multiclasse

# Treinar o modelo com características HOG
model.fit(train_hog_features, train_labels, epochs=5, batch_size=64)

# Avaliar o modelo
test_loss, test_acc = model.evaluate(test_hog_features, test_labels)
print(f'Test accuracy with HOG features: {test_acc}')