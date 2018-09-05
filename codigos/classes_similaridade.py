import glob
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.dummy import Pool as ThreadPool
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from numpy import concatenate as np_concatenate
from numpy import sum as np_sum
from os import path as os_path
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity as sk_cossine

STOP_BR = stopwords.words('portuguese')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

from classes_util import UTIL_MATRIZ, UTIL, UTIL_ARQUIVOS, UTIL_COUNTER, UTIL_TEXTOS
from classes_elastic import ELASTIC


STOP_BR = stopwords.words('portuguese')
REG_MAIUSCULAS = r'^[A-Z\WÁÉÍÓÚÀÂÔÃÕÜÂÔÊÇ]+$'


class UTIL_SIMILARIDADE(object):
    ## Faz o calculo de similaridade baseada no coseno
    @staticmethod
    def cosine_similarity(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[x] * vec2[x] for x in intersection)
        sum1 = sum(vec1[x] ** 2 for x in vec1.keys())
        sum2 = sum(vec2[x] ** 2 for x in vec2.keys())
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        else:
            coef = numerator / denominator
            return coef if coef <= 1 else 1

    @staticmethod
    def _matriz_tf_idf(textos, n_shingles=3):
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, n_shingles), strip_accents='ascii',
                             stop_words=set(STOP_BR))
        matriz_tfidf = tf.fit_transform(textos, ' '.join(textos))
        return matriz_tfidf, tf.vocabulary_

    @staticmethod
    def _matriz_es(textos, campo_elastic=None, incluir_pesos=True, objElastic: ELASTIC = None):
        vet = UTIL_SIMILARIDADE.get_vetores_es(textos=textos, campo=campo_elastic,
                                               objElastic=objElastic, incluir_pesos=incluir_pesos)

        linhas = []
        colunas = []
        valores = []
        termos = {}
        tv = []
        for li, doc in enumerate(vet):
            for term in doc:
                idx = termos.setdefault(term, len(termos))
                valores.append(doc[term])
                linhas.append(li)
                colunas.append(idx)
                tv.append([li, term, doc[term]])
        termos = [t for t in termos]
        csr = csr_matrix((valores, (linhas, colunas)), dtype=float, shape=(len(textos), len(termos)))
        return csr, termos

    @staticmethod
    def vocab(textos, n_shingles=3):
        _, vocab = UTIL_SIMILARIDADE._matriz_tf_idf(textos=textos, n_shingles=n_shingles)
        return vocab

    @staticmethod
    def tfidf(textos, n_shingles=3, nomes=None, incluir_vocab=True):
        matriz_tfidf, vocab = UTIL_SIMILARIDADE._matriz_tf_idf(textos=textos, n_shingles=n_shingles)
        matriz_tfidf = matriz_tfidf.toarray()
        if nomes:
            matriz_tfidf = UTIL_MATRIZ.shift_dir_baixo(matriz=matriz_tfidf, n=1, valor='')
            for i in range(1, len(matriz_tfidf)):
                matriz_tfidf[i, 0] = nomes[i - 1] if i - 1 < len(nomes) else ''
        if incluir_vocab:
            voc = ['' for _ in range(len(vocab))]
            for v in vocab:
                voc[vocab[v]] = v
            for i in range(1, len(matriz_tfidf[0])):
                matriz_tfidf[0, i] = voc[i - 1]
        return matriz_tfidf

    @staticmethod
    def cossenos(textos, n_shingles=3, nomes=None):
        # baseado em https://markhneedham.com/blog/2016/07/27/scitkit-learn-tfidf-and-cosine-similarity-for-computer-science-papers/
        matriz_tfidf, _ = UTIL_SIMILARIDADE._matriz_tf_idf(textos=textos, n_shingles=n_shingles)
        # cossenos = [list(linear_kernel(matriz_tfidf[i:i + 1], matriz_tfidf).flatten()) for i in range(len(textos))]
        cossenos = UTIL_MATRIZ.normaliza(sk_cossine(matriz_tfidf))
        if nomes:
            cossenos = UTIL_MATRIZ.shift_dir_baixo(matriz=cossenos, n=1, valor='')
            for i in range(1, len(cossenos)):
                cossenos[0, i] = nomes[i - 1] if i - 1 < len(nomes) else ''
                cossenos[i, 0] = cossenos[0, i]
        return cossenos

    @staticmethod
    def documentos_matriz(pasta=None, nomes_e_textos=None, textos=None):
        _nomes = None
        if not textos:
            if pasta and len(pasta) > 0:
                _nomes, textos = UTIL_ARQUIVOS.carregar_nomes_e_textos(pasta)
            else:
                _nomes, textos = nomes_e_textos
        if not textos:
            return
        return UTIL_SIMILARIDADE.cossenos(textos=textos, n_shingles=3, nomes=_nomes)

    # Obtem os vetores de termos dos documentos
    @staticmethod
    def get_vetores_es(textos, analyzer=None, campo=None, incluir_pesos=True, objElastic: ELASTIC = None):
        def _getac(vls):
            i, tx = vls
            vetores[i] = objElastic.analyze_counter(analyzer=analyzer, campo=campo, texto=tx, retirar_stop=True)

        vetores = [(i, texto) for i, texto in enumerate(textos)]
        UTIL.map_thread(_getac, vetores, n_threads=5)
        if incluir_pesos:
            counter_pesos = objElastic.term_vector_counter(campo=campo, texto=' '.join(textos), normalizado=True)
            vetores = [UTIL_COUNTER.ajusta_pesos(v, counter_pesos) for v in vetores]

        return vetores

    @staticmethod
    def documentos_matriz_elastic(pasta=None, nomes_e_textos=None, textos=None,
                                  campo_elastic=None, objElastic: ELASTIC = None):
        assert pasta or nomes_e_textos or textos, 'É necessário passar um dos argumentos: pasta, textos ou nomes_e_textos=(nomes,textos)'
        nomes = None
        if not textos:
            if pasta and len(pasta) > 0:
                nomes, textos = UTIL_ARQUIVOS.carregar_nomes_e_textos(pasta)
            else:
                nomes, textos = nomes_e_textos
        # se não tiver dados do elastic, faz com os próprios documentos
        if not (campo_elastic and objElastic):
            return UTIL_SIMILARIDADE.documentos_matriz(pasta=pasta, nomes_e_textos=nomes_e_textos, textos=textos)

        matriz_es, _ = UTIL_SIMILARIDADE._matriz_es(textos=textos, campo_elastic=campo_elastic,
                                                    incluir_pesos=True, objElastic=objElastic)
        cossenos = sk_cossine(matriz_es)
        cossenos = UTIL_MATRIZ.normaliza(cossenos)
        if nomes:
            cossenos = UTIL_MATRIZ.shift_dir_baixo(matriz=cossenos, n=1, valor='')
            for i in range(1, len(cossenos)):
                cossenos[0, i] = nomes[i - 1] if i - 1 < len(nomes) else ''
                cossenos[i, 0] = cossenos[0, i]
        return cossenos

    @staticmethod
    def compara(texto1, texto2):
        mat = UTIL_SIMILARIDADE.documentos_matriz(textos=[texto1, texto2])
        if mat is None:
            return 0
        return mat[0][1]

    @staticmethod
    def compara_elastic(texto1, texto2, campo_elastic=None, objElastic: ELASTIC = None):
        mat = UTIL_SIMILARIDADE.documentos_matriz_elastic(textos=[texto1, texto2],
                                                          campo_elastic=campo_elastic, objElastic=objElastic)
        # UTIL_MATRIZ.print_console(titulo='', matriz=mat)
        if mat is None:
            return 0
        return mat[0][1]

    # retorna o índice do texto, o score e o texto/sentença
    @staticmethod
    def scores_textos(texto_ou_textos, peso_so_maiusculas=2, campo_elastic=None, objElastic: ELASTIC = None, ordem_pesos=True):
        if type(texto_ou_textos) is str:
            textos = UTIL_TEXTOS.sentencas(texto_ou_textos)
        else:
            textos = texto_ou_textos
        if campo_elastic and campo_elastic != '' and objElastic:
            mat, voc = UTIL_SIMILARIDADE._matriz_es(textos=textos, campo_elastic=campo_elastic, objElastic=objElastic)
        else:
            mat, voc = UTIL_SIMILARIDADE._matriz_tf_idf(textos)
        scores = np_sum(mat.toarray(), axis=1)
        res = []
        for i, s in enumerate(scores):
            if peso_so_maiusculas != 1:
                if bool(re.match(REG_MAIUSCULAS, textos[i])):
                    s *= peso_so_maiusculas
            if s>0:
                res.append((i, s, textos[i]))
        if not ordem_pesos:
            return res
        return sorted(res, key=lambda _i: _i[1], reverse=True)

    @staticmethod
    def resumo_textos(texto_ou_textos, min_sentencas=2, min_percentual=5, unir=True, peso_so_maiusculas=2,
                      campo_elastic=None, objElastic: ELASTIC = None):
        scores = UTIL_SIMILARIDADE.scores_textos(texto_ou_textos=texto_ou_textos,
                                                 peso_so_maiusculas=peso_so_maiusculas, campo_elastic=campo_elastic,
                                                 objElastic=objElastic)
        res = []
        s_parcial = 0
        if min_sentencas == 0 and min_percentual == 0:
            min_sentencas = 1
        total = sum(s for _, s, __ in scores)
        for i, s, t in scores:
            s_parcial += s
            res.append(t)
            if len(res) >= min_sentencas and (min_percentual == 0 or 100 * s_parcial / total > min_percentual):
                break
        print('Percentual: ', 100 * s_parcial / total, 'Sentenças: ', len(res), 'Total scores: ',total)
        res = sorted(res, key=lambda i: i[0], reverse=False)
        if unir:
            return ' '.join(res)
        else:
            return res


def alimentar_comparador(pasta, nomes_e_textos=None, objElastic: ELASTIC = None):
    if pasta and len(pasta) > 0:
        nomes, textos = UTIL_ARQUIVOS.carregar_nomes_e_textos(pasta)
    else:
        nomes, textos = nomes_e_textos
    qtd = len(textos)
    print("Arquivos carregados: {}".format(qtd))

    def _grava(v):
        i, n, t = v
        t = UTIL_TEXTOS.limpeza_texto_ascii(t)
        doc = {"Texto_Shingle": t,
               "Texto_Shingle_RAW": t,
               "Texto": t,
               "Atualizacao": objElastic.datahora()}
        objElastic.post_doc(json_data=doc, id=n)
        # print(i)

    l = zip(range(len(nomes)), nomes, textos)
    UTIL.map_thread(_grava, l)


if __name__ == "__main__":

    original = 'Ação trabalhista –ação judicial que envolva pedidos pertinentes à relação de trabalho. Pode ser movida pelo empregado contra a empregador a quem tenha prestado serviço, visando a resgatar direitos decorrentes da relação de emprego, como, também, pode ser de iniciativa do empregador. Usualmente diz-se reclamação trabalhista.'
    parecido = 'Uma ação trabalhista é a ação judicial que envolve alguns pedidos pertinentes à relação de trabalho. Pode ser movida por um empregado contra o seu empregador a quem tenha prestado serviço, visando a resgatar direitos decorrentes da relação de emprego, como, também, pode ser de iniciativa do empregador. Usualmente diz-se reclamação trabalhista.'
    diferente = 'Select files by an absolute date or a relative date by using the /d parameter.'

    elastic = ELASTIC('http://localhost:9200', 'comparador', 'textos')

    print('\nComparando o mesmo texto: ', UTIL_SIMILARIDADE.compara(texto1=original, texto2=original))
    print('\nComparando dois textos próximos: ', UTIL_SIMILARIDADE.compara(texto1=parecido, texto2=original))
    print('\nComparando dois textos diferentes: ', UTIL_SIMILARIDADE.compara(texto1=diferente, texto2=original))

    print('\nComparando o mesmo texto elastic: ',
          UTIL_SIMILARIDADE.compara_elastic(texto1=original, texto2=original,
                                            campo_elastic='Texto_Shingle', objElastic=elastic))
    print('\nComparando dois textos parecidos elastic: ',
          UTIL_SIMILARIDADE.compara_elastic(texto1=parecido, texto2=original,
                                            campo_elastic='Texto_Shingle', objElastic=elastic))
    print('\nComparando dois textos diferentes elastic: ',
          UTIL_SIMILARIDADE.compara_elastic(texto1=diferente, texto2=original,
                                            campo_elastic='Texto_Shingle', objElastic=elastic))

    print('====================================')
    texto = UTIL_ARQUIVOS.carregar_string_arquivo('.\\textos_corpus\\resumir.txt')

    print('== RESUMO SKLEARN')
    print(UTIL_SIMILARIDADE.resumo_textos(texto_ou_textos=texto, min_sentencas=2, min_percentual=0.1))

    print('== RESUMO ELASTIC')
    print(UTIL_SIMILARIDADE.resumo_textos(texto_ou_textos=texto, min_sentencas=2, min_percentual=0.1,
                                          objElastic=elastic, campo_elastic='Texto_Shingle'))

    print('== SCORES SKLEARN')
    [print(s) for s in UTIL_SIMILARIDADE.scores_textos(texto_ou_textos=texto, peso_so_maiusculas=2)[:7]]

    print('== SENTENÇAS')
    texto='Esse texto tem 3 sentenças da Sr(a). e Dra. Maria e o Sr. João é uma. A outra com o art. 333 é a segunda. E por fim, temos fls. 5 a terceira.'
    [print(s) for s in UTIL_TEXTOS.sentencas(texto)]
