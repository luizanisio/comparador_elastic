from collections import Counter

from classes_elastic import ELASTIC
from numpy import concatenate as np_concatenate
from sklearn.metrics.pairwise import cosine_similarity as sk_cossine
from scipy.sparse import csr_matrix
from classes_util import UTIL_MATRIZ, UTIL, UTIL_ARQUIVOS, UTIL_COUNTER, UTIL_TEXTOS

from nltk.corpus import stopwords

STOP_BR = stopwords.words('portuguese')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel


class UTIL_SIMILARIDADE(object):
    ## Faz o calculo de similaridade baseada no coseno
    @staticmethod
    def __matriz_tf_idf__(textos, n_shingles=3):
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, n_shingles), strip_accents='ascii',
                             stop_words=set(STOP_BR))
        matriz_tfidf = tf.fit_transform(textos, ' '.join(textos))
        return matriz_tfidf, tf.vocabulary_

    @staticmethod
    def vocab(textos, n_shingles=3):
        _, vocab = UTIL_SIMILARIDADE.__matriz_tf_idf__(textos=textos, n_shingles=n_shingles)
        return vocab

    @staticmethod
    def tfidf(textos, n_shingles=3, nomes=None, incluir_vocab=True):
        matriz_tfidf, vocab = UTIL_SIMILARIDADE.__matriz_tf_idf__(textos=textos, n_shingles=n_shingles)
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
        matriz_tfidf, _ = UTIL_SIMILARIDADE.__matriz_tf_idf__(textos=textos, n_shingles=n_shingles)
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

        vet = UTIL_SIMILARIDADE.get_vetores_es(textos=textos, campo=campo_elastic,
                                               objElastic=objElastic, incluir_pesos=True)

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
        matriz_es = csr_matrix((valores, (linhas, colunas)), dtype=float, shape=(len(textos), len(termos)))

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
        if mat is None:
            return 0
        return mat[0][1]

###########################################################
###########################################################
## testes dos resultados
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
