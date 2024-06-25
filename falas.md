# 1b e 2b.

**Sobre a camada flatten do modelo HOG MULTICLASSE**
Camada Flatten:

A primeira camada agora é uma Flatten, que achata os dados de entrada para um vetor unidimensional. Isso é adequado para características HOG, que não necessitam de processamento convolucional adicional. O input_shape foi corrigido para refletir a forma esperada das características HOG, que geralmente têm três dimensões (altura, largura, canais) após o processamento HOG.
Os modelos descritos são projetados para trabalhar com características HOG extraídas do conjunto de dados MNIST, mas diferem no tipo de tarefa de classificação para a qual são destinados: um para classificação multiclasse e outro para classificação binária. Vamos analisar cada modelo separadamente e esclarecer a ausência de camadas de aplicação de kernel e pooling.

**Modelo Multiclasse**
Este modelo é projetado para classificar as imagens do MNIST em uma das 10 classes possíveis (dígitos de 0 a 9). O modelo é composto pelas seguintes camadas:

Flatten: Converte os dados de entrada tridimensionais (características HOG) em um vetor unidimensional. A forma de entrada é especificada para corresponder à forma das características HOG extraídas.

Dense (128, activation='relu'): Uma camada densa com 128 neurônios e função de ativação ReLU. Serve para aprender padrões complexos a partir das características HOG achatadas.

Dropout (0.5): Camada de Dropout com taxa de 0.5 para reduzir o risco de overfitting, descartando aleatoriamente 50% das ativações durante o treinamento.

Dense (64, activation='relu'): Outra camada densa, agora com 64 neurônios, também utilizando a função de ativação ReLU.

Dense (10, activation='softmax'): A camada de saída para classificação multiclasse, com 10 neurônios correspondentes às 10 classes do MNIST e a função de ativação softmax, que transforma os logits em probabilidades.

**Modelo Binário**
Este modelo é destinado à classificação binária, ou seja, decidir entre duas classes possíveis. As camadas são:

Dense (128, activation='relu', input_shape=(x_train_hog.shape[1],)): Similar à camada densa do modelo multiclasse, mas especificada diretamente com input_shape para as características HOG. A descrição do input_shape parece ter um erro, pois deveria refletir a dimensionalidade das características HOG achatadas.

Dropout (0.5): Mesma função que no modelo multiclasse, ajudando a prevenir o overfitting.

Dense (64, activation='relu'): Similar à segunda camada densa do modelo multiclasse.

Dense (2, activation='sigmoid'): A camada de saída para classificação binária. Aqui, a escolha da função de ativação sigmoid e duas unidades de saída é incomum. Para classificação binária, geralmente se usa uma única unidade de saída com sigmoid. Duas unidades com sigmoid podem funcionar, mas é menos convencional e pode exigir tratamento especial na interpretação dos resultados e na configuração da função de perda.

**Ausência de Camadas de Kernel e Pooling**
Ambos os modelos não incluem camadas convolucionais (Conv2D) ou de pooling (MaxPooling2D), que são típicas em CNNs para processamento direto de imagens. Isso faz sentido, pois os modelos são projetados para trabalhar com características HOG, que já são um tipo de representação extraída e processada das imagens. As características HOG capturam informações de gradientes e orientações locais, que são úteis para tarefas de visão computacional, incluindo classificação de imagens. Portanto, a aplicação adicional de kernels convolucionais e pooling não é necessária e até mesmo inadequada para esse tipo de dado.

**Conclusão sobre os modelos HOG**
Os modelos fazem sentido para suas respectivas tarefas com o conjunto de dados MNIST processado para extrair características HOG. No entanto, o modelo binário deve ser ajustado para usar uma única unidade de saída com sigmoid para a classificação binária, o que é mais padrão e direto para interpretação e implementação.

# 3.

**Sobre as camadas densas no final do modelo MULTICLASSE BINÁRIA**
Dense(64, activation='relu'): Uma camada densa (ou totalmente conectada) com 64 unidades e função de ativação ReLU. Ela recebe o vetor achatado da camada anterior como entrada.

Dense(1, activation='sigmoid'): A última camada é uma camada densa com uma única unidade e a função de ativação sigmoid. Isso é típico para problemas de classificação binária, onde a saída é a probabilidade de a entrada pertencer a uma das duas classes.

**Sobre a diferença das camadas densas ao final dos modelos que trabalham com os dados BRUTOS**
As duas camadas densas ao final de cada modelo servem a propósitos diferentes, refletindo a natureza das tarefas que cada modelo está tentando resolver: classificação multiclasse e classificação binária.

Função de Ativação Softmax no Modelo Multiclasse:

A função de ativação softmax é utilizada na última camada do modelo de classificação multiclasse. Ela é adequada para este cenário porque a softmax transforma os logits (valores brutos de saída da última camada densa antes da ativação) em probabilidades, garantindo que a soma de todas as probabilidades de classe seja igual a 1. Isso é ideal para classificação multiclasse, onde cada entrada deve ser classificada em uma entre várias classes. No caso do MNIST, existem 10 classes (dígitos de 0 a 9), portanto, a última camada densa tem 10 unidades, uma para cada classe possível.
Função de Ativação Sigmoid no Modelo Binário:

A função de ativação sigmoid é usada na última camada do modelo de classificação binária. Ela é escolhida porque produz uma saída entre 0 e 1, que pode ser interpretada como a probabilidade de a entrada pertencer à classe positiva (ou classe 1). Isso é perfeito para tarefas de classificação binária, onde cada entrada pertence a uma de duas classes possíveis. A última camada densa tem apenas 1 unidade, refletindo a natureza binária da tarefa.
Função de Ativação ReLU na Penúltima Camada:

A função de ativação ReLU (Rectified Linear Unit) é utilizada nas camadas convolucionais e na penúltima camada densa de ambos os modelos. A ReLU é popular em redes neurais profundas devido à sua capacidade de acelerar a convergência do treinamento sem sacrificar a capacidade do modelo de aprender representações complexas. Ela faz isso ao permitir a passagem de valores positivos inalterados, enquanto os valores negativos são zerados. Isso ajuda a mitigar o problema do desaparecimento do gradiente, permitindo que redes mais profundas aprendam efetivamente.
Diferença Fundamental:

A diferença fundamental entre os dois modelos reside na última camada e na função de ativação escolhida, refletindo a natureza da tarefa de classificação que cada modelo está tentando resolver (multiclasse vs. binária). Além disso, a estrutura geral dos modelos é muito semelhante, o que demonstra a flexibilidade das redes convolucionais para lidar com diferentes tipos de tarefas de classificação de imagens, ajustando-se principalmente na camada de saída e na função de ativação correspondente.

**Sobre as camadas densas do modelo MULTICLASSE BINÁRIA**

A primeira camada densa com 128 unidades e ativação ReLU é mantida para aprender padrões complexos nos dados.
Uma camada de Dropout com uma taxa de 0.5 é adicionada para ajudar a prevenir o overfitting, descartando aleatoriamente 50% das ativações durante o treinamento.
Uma segunda camada densa com 64 unidades e ativação ReLU é introduzida para aumentar a capacidade do modelo de aprender representações.
Camada de Saída:

A camada de saída permanece uma camada densa com 10 unidades e ativação softmax, adequada para a classificação multiclasse do MNIST, onde cada unidade corresponde a uma das 10 classes possíveis (dígitos de 0 a 9).
Considerações Finais:

Este modelo é mais simples e direto, focado em aprender a partir das características HOG sem tentar aplicar operações convolucionais ou de pooling, que são menos relevantes para dados já processados e resumidos.
A inclusão de uma camada de Dropout ajuda a mitigar o risco de overfitting, especialmente importante quando se trabalha com um conjunto de características que pode ser de alta dimensão, como é frequentemente o caso com HOG.
Ajustar o número de unidades nas camadas densas e a taxa de dropout pode ser necessário dependendo do desempenho específico do modelo no conjunto de dados.

# 4. Lógica do treinamento da estrutura completa.

Para analisar a lógica do treinamento dos modelos de CNNs com o conjunto de dados MNIST, considerando as variações (dados brutos vs. extração de características HOG e multiclasse vs. classes binárias), podemos seguir os seguintes passos:

Preparação dos Dados:

Dados Brutos: Carregar o conjunto de dados MNIST diretamente e normalizar os valores dos pixels para o intervalo [0, 1]. Redimensionar as imagens para o formato adequado para a CNN, que geralmente inclui a adição de um canal de cor (mesmo para imagens em escala de cinza).
Extração de Características HOG: Utilizar um extrator de características HOG para transformar as imagens em um conjunto de características que descrevem a orientação dos gradientes ou bordas das imagens. Isso pode ajudar a reduzir a dimensionalidade dos dados e focar em aspectos importantes para a classificação.
Conversão dos Rótulos:

Multiclasse: Converter os rótulos para o formato one-hot encoding, que é necessário para a classificação multiclasse em redes neurais.
Classes Binárias: Para tarefas com duas classes, os rótulos podem ser ajustados para representar as duas classes de interesse, possivelmente também utilizando one-hot encoding ou mantendo-os como 0 e 1.
Construção do Modelo:

Utilizar camadas convolucionais (Conv2D) e de pooling (MaxPooling2D) para extrair características espaciais das imagens. Camadas como Dropout e Dense são usadas para evitar overfitting e para a classificação final.
A diferença principal entre os modelos para tarefas multiclasse e binárias estará na última camada Dense, onde o número de unidades deve corresponder ao número de classes (10 para multiclasse no MNIST e 2 para binário) e a função de ativação (geralmente softmax para multiclasse e sigmoid para binário).
Compilação do Modelo:

Definir o otimizador (ex.: Adam), a função de perda (categorical_crossentropy para multiclasse e binary_crossentropy para binário), e as métricas (usualmente accuracy).
Treinamento do Modelo:

Treinar o modelo com os dados preparados, ajustando os hiperparâmetros conforme necessário (taxa de aprendizado, número de épocas, tamanho do batch, etc.).
Avaliação e Ajustes:

Avaliar o desempenho do modelo nos dados de teste e fazer ajustes conforme necessário, como alterar a arquitetura do modelo, os hiperparâmetros ou a forma como os dados são preparados.
Salvamento dos Hiperparâmetros e Pesos:

Para reprodução e análise futura, salvar os hiperparâmetros e os pesos iniciais do modelo pode ser útil.
A estrutura completa do treinamento envolve a preparação cuidadosa dos dados, a construção e ajuste fino da arquitetura do modelo, e a avaliação rigorosa do desempenho do modelo. A escolha entre usar dados brutos ou características HOG, assim como a definição do problema como multiclasse ou binário, influenciará principalmente as etapas de preparação dos dados e a configuração final do modelo.

# 5. Critério de parada do treinamento.

Não foi implementado. Estamos testando com poucas épocas

# 6. Procedimentos de cálculo de erro na camada de saída.

**Categorical_crossentropy - Utilizada nos arquivos que trabalham com multiclasses**
A função categorical_crossentropy é usada como uma função de perda (loss function) em problemas de classificação onde as classes são mutuamente exclusivas. Em outras palavras, é usada quando cada amostra pertence exatamente a uma classe. Aqui está um passo a passo de como ela funciona:

Entrada: A função recebe como entrada as probabilidades previstas pelo modelo para cada classe (geralmente após uma camada softmax) e as classes verdadeiras em formato one-hot encoding.
Cálculo do Logaritmo: Para cada amostra, o logaritmo natural da probabilidade prevista correspondente à classe verdadeira é calculado.
Negativo: O valor logarítmico é então multiplicado por -1, tornando-o positivo (já que o logaritmo de um número entre 0 e 1 é negativo).
Média: A média desses valores é calculada para todas as amostras no lote para obter o valor final da perda.
A ideia é minimizar essa perda durante o treinamento, o que significa que o modelo está melhorando suas previsões para se alinhar com as classes verdadeiras. Quanto menor a perda, melhor o modelo em prever a classe correta.

Ver mais detalhes em: https://www.tensorflow.org/api_docs/python/tf/keras/losses/categorical_crossentropy

**Binary_crossentropy - Utilizada nos arquivos que trabalham com classes binárias**
A função binary_crossentropy é usada como uma função de perda (loss function) em problemas de classificação binária, onde existem apenas duas classes possíveis. Ela mede o desempenho de um modelo de classificação cuja saída é um valor de probabilidade entre 0 e 1. A função binary_crossentropy é adequada para problemas onde você precisa prever se algo é verdadeiro ou falso, sim ou não, etc. Aqui está um passo a passo de como ela funciona:

Entrada: A função recebe como entrada as probabilidades previstas pelo modelo para a classe positiva (geralmente após uma camada sigmoid) e os rótulos verdadeiros, que devem ser 0 ou 1.
Cálculo do Logaritmo: Para cada amostra, calcula-se o logaritmo da probabilidade prevista se o rótulo verdadeiro for 1, e o logaritmo de (1 - probabilidade prevista) se o rótulo verdadeiro for 0.
Multiplicação: Esses logaritmos são multiplicados pelos rótulos verdadeiros correspondentes (ou 1 menos o rótulo verdadeiro, no caso de rótulos 0), o que efetivamente seleciona o logaritmo correto para cada caso.
Negativo: Os valores resultantes são negativos, pois o logaritmo de um número entre 0 e 1 é negativo.
Média: A média desses valores negativos é calculada para todas as amostras no lote para obter o valor final da perda.
O objetivo é minimizar essa perda, o que indica que o modelo está se tornando melhor em prever a classe correta. Uma perda menor significa que as probabilidades previstas pelo modelo estão mais próximas dos rótulos verdadeiros.

Ver mais detalhes em: https://www.tensorflow.org/api_docs/python/tf/keras/losses/binary_crossentropy

# 7. Procedimento de cálculo da resposta da rede em termos reconhecimento de caractere.

**Brutos Multiclasse (ou sobre os dados brutos [multiclasse/binário])**

Procedimento de Cálculo da Resposta da Rede

A rede é treinada para classificar imagens de dígitos manuscritos, passando por um processo de normalização, convolução, pooling, flattening e classificação com funções de ativação apropriadas e um processo de otimização para ajustar os pesos da rede com base na função de perda.

Forward Pass:

- Cada imagem de entrada passa pelas camadas convolucionais, onde os filtros convolucionais extraem características importantes.
- As camadas de pooling reduzem a dimensionalidade das características extraídas.
- As camadas densas processam as características extraídas e fazem a classificação.

Função de Ativação:

- Funções de ativação ReLU (Rectified Linear Unit) são usadas nas camadas convolucionais e densas intermediárias.
- A função de ativação softmax na última camada densa converte as saídas em probabilidades para cada uma das classes (dígitos de 0 a 9).

Cálculo da Perda e Atualização dos Pesos:

- A função de perda categorical_crossentropy calcula a diferença entre a previsão do modelo e a verdade real.
- O otimizador Adam ajusta os pesos da rede para minimizar a perda calculada.

**Hog Binário (ou sobre os dados hog [multiclasse/binário])**

Procedimento de Cálculo da Resposta da Rede

Extração de Características HOG:

- Cada imagem passa pelo descritor HOG, que calcula histogramas de gradientes orientados para capturar a estrutura local das bordas.

Forward Pass:

- As características HOG normalizadas são alimentadas na rede neural.
- A rede neural consiste em uma camada densa oculta com 128 neurônios e função de ativação ReLU, seguida de uma camada de saída com 2 neurônios e função de ativação softmax para classificação binária.

Função de Ativação:

- ReLU é usada na camada oculta para introduzir não-linearidades.
- Softmax é usada na camada de saída para converter as saídas em probabilidades para as duas classes.

Cálculo da Perda e Atualização dos Pesos:

- A função de perda categorical_crossentropy calcula a diferença entre as previsões do modelo e os rótulos reais.
- O otimizador Adam ajusta os pesos da rede para minimizar a perda calculada.

# 8. Teste da CNN para o conjunto de dados MNIST.

Apresentar os 4 testes

# 9. Procedimento de cálculo da matriz de confusão

A matriz de confusão é calculada da seguinte maneira:

1. As previsões do modelo são obtidas usando o método predict no conjunto de teste x_test. Isso resulta em um array de saídas, onde cada saída é um vetor de probabilidades para as classes.
2. As classes previstas são determinadas usando a função np.argmax(saidas, axis=1), que seleciona os índices dos valores máximos ao longo do eixo 1 (classes) das saídas previstas, convertendo as probabilidades em previsões de classe específicas.
3. As classes verdadeiras são extraídas de y_test usando a mesma função np.argmax(y_test, axis=1), que seleciona os índices dos valores máximos ao longo do eixo 1, representando as classes verdadeiras.
4. A função confusion_matrix da biblioteca sklearn.metrics é usada para calcular a matriz de confusão, passando as classes verdadeiras e as classes previstas como argumentos.
5. A matriz de confusão resultante é visualizada usando a biblioteca seaborn com a função heatmap, que plota a matriz de confusão com os valores de cada célula anotados e um mapa de cores para representação visual.

```python
# Previsões

classes_previstas = np.argmax(saidas, axis=1)
classes_verdadeiras = np.argmax(y_test, axis=1)

# Calcule a matriz de confusão

cm = confusion_matrix(classes_verdadeiras, classes_previstas)

# Plot a matriz de confusão

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Matriz de Confusão')
plt.show()
```
