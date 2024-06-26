# Integrantes
# Elisa Yea Ji Lee - NUSP 11892473
# João Victor Andrade Lúcio - NUSP 11207877
# Leonardo Zoppello Fernandes - NUSP 13838749
# Marcos Paulo Tomás Ferreira - NUSP 13747950
# Rafael Moura de Almeida - NUSP 11225505
# Thomas Delfs - NUSP 13837175


import h5py
import sys

def imprimir_dados_recursivamente(objeto, prefixo=''):
    """
    Função recursiva para imprimir os dados de todos os conjuntos de dados em um arquivo HDF5,
    expandindo todos os grupos e subgrupos.
    """
    for nome in objeto:
        item = objeto[nome]
        nome_completo = f'{prefixo}/{nome}' if prefixo else nome  # Constrói o nome completo do caminho
        if isinstance(item, h5py.Dataset):
            # Imprime os dados do conjunto de dados
            dados = item[()]
            print(f'{nome_completo}: {dados}')
        elif isinstance(item, h5py.Group):
            # Se for um grupo, chama a função recursivamente para seus subitens
            print(f'{nome_completo}: Grupo')
            imprimir_dados_recursivamente(item, nome_completo)

def imprimir_conteudo_arquivo(caminho_arquivo):
    try:
        with h5py.File(caminho_arquivo, 'r') as arquivo:
            imprimir_dados_recursivamente(arquivo)
    except IOError:
        print(f'Erro ao abrir o arquivo "{caminho_arquivo}".')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <caminho_para_o_arquivo>")
    else:
        caminho_arquivo = sys.argv[1]
        imprimir_conteudo_arquivo(caminho_arquivo)
