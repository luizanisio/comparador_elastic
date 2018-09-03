import logging
from classes_util import UTIL_MATRIZ, UTIL, UTIL_ARQUIVOS, UTIL_COUNTER, UTIL_TEXTOS
from classes_elastic import ELASTIC
from classes_similaridade import UTIL_SIMILARIDADE

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

def alimentar_comparador(pasta, nomes_e_textos=None, objElastic: ELASTIC = None):
    if pasta and len(pasta) > 0:
        nomes, textos = UTIL_ARQUIVOS.carregar_nomes_e_textos(pasta)
    else:
        nomes, textos = nomes_e_textos
    qtd = len(textos)
    print("Arquivos carregados: {}".format(qtd))

    UTIL.map_thread(UTIL_TEXTOS.corrigir_simbolos, textos)

    def grava(v):
        i, n, t = v
        doc = {"Texto_Shingle": t,
               "Texto_Shingle_RAW": t,
               "Texto": t,
               "Atualizacao": objElastic.datahora()}
        objElastic.post_doc(json_data=doc, id=n)
        # print(i)

    nm_tx = zip(range(len(nomes)), nomes, textos)
    UTIL.map_thread(grava, nm_tx)
    print('Documentos gravados: ', len(nomes))



if __name__ == '__main__':

    elastic = ELASTIC('http://localhost:9200', 'comparador', 'textos')
    #alimentar_comparador(pasta='.\\textos_corpus\\',objElastic=elastic)


    pasta = '.\\textos_comp\\'
    cos_es = UTIL_SIMILARIDADE.documentos_matriz_elastic(pasta=pasta,
                                                         campo_elastic='Texto_Shingle_RAW',
                                                         objElastic=elastic)
    if cos_es is not None:
        UTIL_MATRIZ.gravar(arquivo='similaridade_es.txt', matriz=cos_es, espacar=True)
        print('Arquivo "similaridade_es.txt" gravado com sucesso!')
    else:
        print('Nenhum documento encontrado em "{}"!'.format(pasta))

    print('')
    matriz = UTIL_SIMILARIDADE.documentos_matriz(pasta=pasta)
    if matriz is not None:
        UTIL_MATRIZ.gravar(arquivo='similaridade.txt', matriz=matriz, espacar=True)
        print('Arquivo "similaridade.txt" gravado com sucesso!')
    else:
        print('Nenhum documento encontrado em "{}"!'.format(pasta))
