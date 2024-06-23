# Importar Bibliotecas Necessárias
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from skimage.feature import hog
from skimage import exposure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar o Conjunto de Dados MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#############################################################################################################
############################################## DEFINIÇÃO - HOG ##############################################
#############################################################################################################
# Função para Extração de Características HOG
def extract_hog_features(images):
    hog_features = []
    for image in images:
        # Assegura que a imagem está em 2D (altura, largura)
        if image.ndim == 3 and image.shape[2] == 1:  # Para imagens com um eixo de canal explícito
            image = image.reshape(image.shape[0], image.shape[1])
        fd = hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
        hog_features.append(fd)
    return np.array(hog_features)

# Extração de Características HOG
x_train_hog = extract_hog_features(x_train)
x_test_hog = extract_hog_features(x_test)

# Normalização das Características HOG
scaler = StandardScaler().fit(x_train_hog)
x_train_hog_norm = scaler.transform(x_train_hog)
x_test_hog_norm = scaler.transform(x_test_hog)


#############################################################################################################
################################################ MULTICLASSE ################################################
#############################################################################################################

################################################## BRUTOS ###################################################

# Pré-processamento dos Dados para Tarefa Multiclasse
x_train_mc = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test_mc = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train_mc = to_categorical(y_train, 10)
y_test_mc = to_categorical(y_test, 10)

# Construção da Rede Neural Convolutiva (CNN) para Tarefa Multiclasse
model_mc = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model_mc.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 1. Salvar Hiperparâmetros
hiperparametros = {
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "metrics": ["accuracy"],
    "epochs": 5,
    "batch_size": 64
}
with open("hiperparametros.json", "w") as hp_file:
    json.dump(hiperparametros, hp_file)

# 2. Salvar Pesos Iniciais
model_mc.save_weights("pesos_iniciais.weights.h5")

# 3. Callback para Salvar o Erro por Iteração
class CustomCallback(Callback):
    def on_train_begin(self, logs=None):
        self.erros = []

    def on_epoch_end(self, epoch, logs=None):
        self.erros.append(logs.get('loss'))

    def on_train_end(self, logs=None):
        np.save("erro_por_iteracao.npy", self.erros)

# Treinamento da CNN com Dados MNIST Brutos para Tarefa Multiclasse
model_mc.fit(x_train_mc, y_train_mc, validation_data=(x_test_mc, y_test_mc), epochs=10, batch_size=64, callbacks=[CustomCallback()])

# 5. Salvar Pesos Finais
model_mc.save_weights("pesos_finais.weights.h5")

test_loss_mc, test_acc_mc  = model_mc.evaluate(x_test_mc, y_test_mc)

# 6. Salvar Saídas para Dados de Teste
saidas = model_mc.predict(x_test_mc)
np.save("saidas_teste.npy", saidas)


#################################################### HOG ####################################################
# Redefinir a CNN para aceitar entradas HOG (ajustar a entrada conforme necessário)
model_hog = Sequential([
    Dense(128, activation='relu', input_shape=(x_train_hog_norm.shape[1],)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model_hog.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento da CNN com Características HOG para Tarefa Multiclasse
model_hog.fit(x_train_hog_norm, y_train_mc, validation_data=(x_test_hog_norm, y_test_mc), epochs=10)
test_loss_mc_hog, test_acc_mc_hog = model_hog.evaluate(x_test_hog_norm, y_test_mc)


#############################################################################################################
########################################### CLASSIFICAÇÃO BINÁRIA ###########################################
#############################################################################################################

#################################################### HOG ####################################################
# Preparação dos Dados para Tarefa de Classificação Binária (exemplo: classes 0 e 1)
# Selecionar dados das classes 0 e 1
binary_classes = [0, 1]
idx = np.where((y_train == binary_classes[0]) | (y_train == binary_classes[1]))[0]
x_train_bin, y_train_bin = x_train_hog_norm[idx], y_train[idx]
idx = np.where((y_test == binary_classes[0]) | (y_test == binary_classes[1]))[0]
x_test_bin, y_test_bin = x_test_hog_norm[idx], y_test[idx]

# Converter rótulos para formato binário
y_train_bin = (y_train_bin == binary_classes[1]).astype(int)
y_test_bin = (y_test_bin == binary_classes[1]).astype(int)

# Construir e Treinar a CNN para Tarefa de Classificação Binária
model_bin = Sequential([
    Dense(128, activation='relu', input_shape=(x_train_hog_norm.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_bin.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento da CNN com Características HOG para Tarefa Binária
model_bin.fit(x_train_bin, y_train_bin, validation_data=(x_test_bin, y_test_bin), epochs=10)
test_loss_bin_hog, test_acc_bin_hog  = model_bin.evaluate(x_test_bin, y_test_bin)

#############################################################################################################
################################################# RESULTADOS ################################################
#############################################################################################################

print(
    f"Test accuracy (Multiclass): {test_acc_mc}, Test loss (Multiclass): {test_loss_mc}\n"
    f"Test accuracy HOG (Multiclass): {test_acc_mc_hog}, Test loss HOG (Multiclass): {test_loss_mc_hog}\n"
    f"Test accuracy HOG (Binary): {test_acc_bin_hog}, Test loss HOG (Binary): {test_loss_bin_hog}"
)

# 1. Imprimir conteúdo de um arquivo .npy
def imprimir_npy(filepath):
    data = np.load(filepath)
    print(data)

# 2. Imprimir pesos de um arquivo .weights.h5
def imprimir_weights_h5(filepath):
    # Defina a mesma arquitetura do modelo usada para salvar os pesos
    model = Sequential([
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.load_weights(filepath)
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)

# 3. Imprimir conteúdo de um arquivo .json
def imprimir_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        print(data)

# Exemplo de uso
imprimir_json('hiperparametros.json')
imprimir_npy('erro_por_iteracao.npy')
# imprimir_weights_h5('pesos_iniciais.weights.h5')
# imprimir_weights_h5('pesos_finais.weights.h5')
imprimir_npy('saidas_teste.npy')