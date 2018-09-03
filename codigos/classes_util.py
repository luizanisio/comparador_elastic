import codecs
import re
import math
from heapq import nlargest
from collections import defaultdict, Counter
import glob
import os.path as os_path
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.dummy import Pool as ThreadPool
from numpy import vstack as np_vstack
from numpy import array as np_array

class UTIL_ARQUIVOS(object):
    @staticmethod
    def carregar_string_arquivo(arq):
        return UTIL_ARQUIVOS.carregar_arquivo(arq=arq, limpar=True, juntar_linhas=True, retornar_tipo=False)

    @staticmethod
    def carregar_lista_arquivo(arq):
        return UTIL_ARQUIVOS.carregar_arquivo(arq=arq, limpar=True, juntar_linhas=False, retornar_tipo=False)

    @staticmethod
    def carregar_arquivo(arq, limpar=False, juntar_linhas=False, retornar_tipo=False):
        tipos = ['utf8', 'ascii', 'latin1']
        linhas = None
        tipo = None
        for tp in tipos:
            try:
                with open(arq, encoding=tp) as f:
                    tipo, linhas = (tp, f.read().splitlines())
                    break
            except UnicodeError:
                continue
        if not linhas:
            with open(arq, encoding='latin1', errors='ignore') as f:
                tipo, linhas = ('latin1', f.read().splitlines())
        # otimiza os tipos de retorno
        if limpar and juntar_linhas:
            linhas = re.sub('\s+\s*', ' ', ' '.join(linhas))
        elif limpar:
            linhas = [re.sub('\s+\s*', ' ', l) for l in linhas]
        elif juntar_linhas:
            linhas = ' '.join(linhas)
        if retornar_tipo:
            return tipo, linhas
        else:
            return linhas

    # retorna a lista de (arq, texto)
    @staticmethod
    def carregar_nomes_e_textos(pasta='.\\', nm_reduzido=True):
        arqs = [f for f in glob.glob(r"{}*.txt".format(pasta))]
        #print('Arquivos das pastas', pasta, arqs)
        nms = ['' for _ in range(len(arqs))]
        txts = ['' for _ in range(len(arqs))]

        def _lerarq(i):
            txts[i] = UTIL_ARQUIVOS.carregar_arquivo(arq=arqs[i], juntar_linhas=True, retornar_tipo=False)
            if nm_reduzido:
                _, a = os_path.split(arqs[i])
            nms[i] = a

        with ThreadPoolExecutor(max_workers=10) as executor:
            for i in range(len(arqs)):
                executor.submit(_lerarq, i)
        # print('textos: ',nms)
        return nms, txts

    # Recebe o nome de uma pasta ou um array de pastas e cria o arqZip com o conteúdo dela(s)
    @staticmethod
    def zipdir(diretorio, arqZip):
        def _zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file))

        if not diretorio or len(diretorio) == 0:
            return
        zipf = zipfile.ZipFile(arqZip, 'w', zipfile.ZIP_DEFLATED)
        # Zipa a pasta ou a lista de pastas enviadas
        if isinstance(diretorio, str):
            _zipdir(diretorio, zipf)
        else:
            [_zipdir(d, zipf) for d in diretorio]
        zipf.close()


class UTIL_MATRIZ(object):
    @staticmethod
    def gravar(arquivo, matriz, espacar=False):
        fout = open(arquivo, "w", encoding="utf8")
        q = 0
        if espacar and len(matriz) > 0 and len(matriz[0]) > 0:
            q = max([max([len(str(col)) for col in lin]) for lin in matriz])
        for linha in matriz:
            if not espacar:
                linha = '\t'.join([str(col) for col in linha])
            else:
                linha = '\t'.join([str(col).ljust(q) for col in linha])
            fout.write(linha + '\n')
        fout.close()

    @staticmethod
    def criar(linhas=1, colunas=1, valor=None):
        return [[valor for _ in range(colunas)] for _ in range(linhas)]

    @staticmethod
    def shift_dir_baixo(matriz, n=1, valor=None):
        # inclui n linhas e colunas à esquerda da matriz
        for i in range(n):
            linhas = len(matriz)
            m_colunas = [valor for _ in range(linhas)]
            nova = np_vstack((np_array(m_colunas), np_array(matriz).T)).T
            m_linhas = [valor for _ in range(len(nova[0]))]
            matriz = np_vstack((np_array(m_linhas), nova))
        return matriz

    @staticmethod
    def print_console(titulo, matriz):
        if titulo and len(titulo) > 0:
            print(titulo)
        q = max([max([len(str(col)) for col in lin]) for lin in matriz])
        print('\n'.join(['\t'.join([str(col).ljust(q) for col in lin]) for lin in matriz]))

    @staticmethod
    def normaliza(matriz):
        minv = min([min([co for co in li]) for li in matriz])
        maxv = max([max([co for co in li]) for li in matriz])
        divisor = maxv - minv
        novo = []
        for li in matriz:
            novo.append([1 if divisor == 0 else minv + ((co - minv) / divisor) for co in li])
        return novo


class UTIL_COUNTER(object):
    # Normaliza os valores de pesos de um counter entre 1 e 2 para multiplicação
    @staticmethod
    def normaliza_01(counter):
        if not counter:
            return Counter({})
        # normaliza os pesos >> 1+peso/total de pesos
        maxV = max(counter.values())
        minV = min(counter.values())
        if maxV == minV:
            return counter
        return Counter({k: (counter.get(k, 1) - minV) / (maxV - minV) for k in counter.keys()})

    @staticmethod
    def ordena(counter, reverso=False):
        return sorted(Counter(counter).items(), key=lambda i: i[1], reverse=reverso)

    # Multiplica um counter de termos por um de pesos
    @staticmethod
    def ajusta_pesos(counter_origem, counter_pesos):
        if not counter_origem:
            return Counter({})
        if not counter_pesos:
            return counter_origem
        return Counter({k: counter_origem.get(k, 1) * counter_pesos.get(k, 1) for k in counter_origem.keys()})


class UTIL(object):
    @staticmethod
    def progress_bar(current_value, total, msg=''):
        increments = 50
        percentual = int((current_value / total) * 100)
        i = int(percentual // (100 / increments))
        text = "\r[{0: <{1}}] {2:.2f}%".format('=' * i, increments, percentual)
        print('{} {}           '.format(text, msg), end="\n" if percentual == 100 else "")

    @staticmethod
    def map_thread(func, lista, n_threads=5):
        # print('Iniciando {} threads'.format(n_threads))
        pool = ThreadPool(n_threads)
        pool.map(func, lista)
        pool.close()
        pool.join()
        # print('Finalizando {} threads'.format(n_threads))


class UTIL_TEXTOS(object):
    @staticmethod
    def corrigir_simbolos(texto):
        t = re.sub(r"[´`“”\"“”" + u"\u2018" + u"\u2019" + "]", "'", texto)
        t = re.sub(r"[°º" + u"\u2022" + "]", "º", t)
        return re.sub(r"[–—" + u"\u2026" + "]", "-", t)

    ## Limpeza do texto
    @staticmethod
    def limpeza_texto_ascii(texto):
        return re.sub('\s+', ' ',
                      normalize('NFKD', texto)
                      .encode('ASCII', 'ignore')
                      .decode('ASCII')
                      ).lower().strip()

    @staticmethod
    def removeacentos(texto):
        txt = re.sub(r"\s+", " ", texto).lower()
        acentos = [('áâàãä', 'a'),
                   ('éèêë', 'e'),
                   ('íìîï', 'i'),
                   ('óòôöõ', 'o'),
                   ('úùüû', 'u'),
                   ('ç', 'c'),
                   ('ñ', 'n')]
        for de, para in acentos:
            txt = re.sub(r"[{}]".format(de), para, txt)
        return txt


if __name__ == "__main__":
    # main('resumir.txt',5)
    print('===============================================')
