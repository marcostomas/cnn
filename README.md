# Convolutional Neural Network (CNN)

Criação de uma CNN para classificação binária e multiclasse usando o Keras. Este é o segundo trabalho apresentado para a disciplina de Inteligência Artificial (ACH2016) do 1º semestre de 2024.

## Integrantes

- Elisa Yea Ji Lee - NUSP 11892473
- João Victor Andrade Lúcio - NUSP 11207877
- Leonardo Zoppello Fernandes - NUSP 13838749
- Marcos Paulo Tomás Ferreira - NUSP 13747950
- Rafael Moura de Almeida - NUSP 11225505
- Thomas Delfs - NUSP 13837175

## Descrição

O trabalho está divido em quatro notebooks Jupyter em duas pastas `binaria/` e `multiclasse/`.

Por exemplo, dentro da pasta `binaria/` estão as duas redes que trabalharão com os dados brutos e com extrator hog para classes binárias. O mesmo vale para o outro diretório.

O arquivo `imprimir_dados_weights_h5.py` serve para imprimir no terminal os dados dos pesos iniciais e finais, já que os arquivos de pesos estão codificados em binário. Para imprimir os pesos no terminal, rode:

```
<py|python> <imprimir_dados_weights_h5.py> <caminho_arquivo_de_pesos>
```

O arquivo recebe como argumento da linha de comando o caminho do arquivo de pesos (de extensão .weights.h5).
Não forneça mais de um arquivo de pesos por vez. Execute py em ambiente windows e python em ambientes unix ou unix-like.

## Bibliotecas necessárias

Verifique as instruções de instalação no site de cada biblioteca.

- [Tensorflow](https://www.tensorflow.org/?hl=pt-br)
- [NumPy](https://numpy.org/)
- [SciKit-Learn](https://scikit-learn.org/stable/)
- [SciKit-Image](https://scikit-image.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
