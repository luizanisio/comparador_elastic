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
        # print('Arquivos das pastas', pasta, arqs)
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


ABREVIACOES = ['sra?s?', 'exm[ao]s?', 'ns?', 'nos?', 'doc', 'ac', 'publ', 'ex', 'lv', 'vlr?', 'vls?',
               'exmo(a)', 'ilmo(a)', 'av', 'of', 'min', 'livr?', 'co?ls?', 'univ', 'resp', 'cli', 'lb',
               'dra?s?', '[a-z]+r\(as?\)', 'ed', 'pa?g', 'cod', 'prof', 'op', 'plan', 'edf?', 'func', 'ch',
               'arts?', 'artigs?', 'artg', 'pars?', 'rel', 'tel', 'res', '[a-z]', 'vls?', 'gab', 'bel',
               'ilm[oa]', 'parc', 'proc', 'adv', 'vols?', 'cels?', 'pp', 'ex[ao]', 'eg', 'pl', 'ref',
               '[0-9]+', 'reg', 'f[ilí]s?', 'inc', 'par', 'alin', 'fts', 'publ?', 'ex', 'v. em', 'v.rev']

ABREVIACOES_RGX = re.compile(r'(?:{})\.\s*$'.format('|\s'.join(ABREVIACOES)), re.IGNORECASE)

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

    @staticmethod
    def sentencas(texto, min_len=5):
        # baseado em https://stackoverflow.com/questions/25735644/python-regex-for-splitting-text-into-sentences-sentence-tokenizing
        texto = re.sub(r'\s\s+', ' ', texto)
        EndPunctuation = re.compile(r'([\.\?\!]\s+)')
        # print(NonEndings)
        parts = EndPunctuation.split(texto)
        sentencas = []
        sentence = []
        for part in parts:
            txt_sent = ''.join(sentence)
            q_len = len(txt_sent)
            if len(part) and len(sentence) and q_len >= min_len and \
                    EndPunctuation.match(sentence[-1]) and \
                    not ABREVIACOES_RGX.search(txt_sent):
                sentencas.append(txt_sent)
                sentence = []

            if len(part):
                sentence.append(part)
        if sentence:
            sentencas.append(''.join(sentence))
        return sentencas

    @staticmethod
    def possiveis_abreviacoes(texto, min_len=4):
        NAO_ABREVIACOES = ['mais', 'cit', 'stf', 'quo', 'mm', 'voto', 'cpc', 'etc', 'sede', 'df', 'caso', 'tse', 'casu',
                           'data', 'ale', 'irt', 'xi', 'li', 'oro', 'des', 'ih', 'lvi', 'ii', 'neto', 'sk', 'sark',
                           'ou', 'disp', 'def', 'come', 'taf', 'dc', 'on', 'licc', 'leis', 'ai', 'pois', 'edif', 'ft',
                           'cpis', 'trf', 'aer', 'mar', 'to', 'cj', 'real', 'vi', 'in', 'limo', 'pis', 'xii', 'tes',
                           'tema', 'de', 'ctn', 'rt', 'como', 'doe', 'oct', 'olk', 'las', 'cd', 'ltda', 'voor', 'oot',
                           'que', 'nal', 'el', 'um', 'gg', 'erro', 'desa', 'rte', 'iv', 'ipc', 'mini', 'ari', 'uy',
                           'la', 'lide', 'atos', 'wnt', 'vs', 'julg', 'base', 'wn', 'ir', 'nyy', 'low', 'rs', 'jr',
                           'is', 'rei', 'dipp', 'fed', 'eles', 'rec', 'pgr', 'crt', 'lf', 'tt', 'nr', 'em', 'jogo',
                           'dano', 'do', 'grau', 'cpmf', 'pr', 'col', 'tri', 'bem', 'todo', 'fta', 'lxix', 'dias',
                           'suai', 'ct', 'fe', 'tal', 'az', 'ni', 'fr', 'writ', 'urv', 'adct', 'il', 'vokl', 'ag',
                           'tela', 'ick', 'iii', 'dra', 'ano', 'it', 'dj', 'stj', 'ila', 'lima', 're', 'sa', 'cc',
                           'bela', 'jj', 'tipo', 'ta', 'fla', 'ri', 'tx', 'rosa', 'hoje', 'nem', 'bl', 'mora', 'dec',
                           'mas', 'ine', 'md', 'he', 'oab', 'remo', 'est', 'rum', 'pena', 'nula', 'ato', 'stip', 'di',
                           'ob', 'ofr', 'ltd', 'ter', 'be', 'dl', 'com', 'ros', 'ill', 'nae', 'cfr', 'fia', 'sum',
                           'vdos', 'ke', 'fli', 'rtz', 'edci', 'niko', 'fti', 'ut', 'runc', 'iff', 'pift', 'esmo',
                           'ento', 'ffs', 'dolo', 'vido', 'dihm', 'id', 'enm', 'ic', 'dada', 'vir', 'tl', 'ar', 'vel',
                           'dez', 'gisi', 'en', 'sc', 'tora', 'dd', 'nova', 'ase', 'ttjl', 'tyk', 'ot', 'os', 'suo',
                           'fala', 'te', 'ria', 'ie', 'tc', 'nade', 'tcu', 'ela', 'shs', 'swe', 'dita', 'siam', 'cr',
                           'aro', 'eis', 'modo', 'cabo', 'este', 'ttir', 'rn', 'cs', 'agu', 'deve', 'oa', 'rep', 'rwa',
                           'fora', 'oh', 'sup', 'sia', 'seja', 'vo', 'dele', 'aw', 'vd', 'ma', 'cruz', 'tj', 'bi',
                           'abr', 'pim', 'andr', 'arr', 'pkt', 'nac', 'tare', 'pura', 'ib', 'irpj', 'suga', 'zti', 'jq',
                           'tf', 'oest', 'ele', 'gen', 'wk', 'ectt', 'mi', 'mane', 'omin', 'vv', 'cone', 'boa', 'paga',
                           'frk', 'melo', 'jun', 'iy', 'dal', 'isso', 'clt', 'rr', 'meta', 'das', 'sim', 'ord', 'crfb',
                           'xiv', 'stir', 'et', 'dei', 'kl', 'oco', 'anos', 'nane', 'qd', 'cos', 'incs', 'foi', 'mg',
                           'caa', 'ss', 'piam', 'lk', 'edil', 'ye', 'fux', 'st', 'iva', 'oe', 'vis', 'fab', 'si',
                           'puc', 'ia', 'fato', 'lvii', 'isto', 'dom', 'pais', 'cep', 'ds', 'ails', 'an', 'itci',
                           'jus', 'tari', 'sn', 'acre', 'ao', 'iik', 'mb', 'tu', 'ekmo', 'cimo', 'mano', 'pcat',
                           'adin', 'isi', 'ito', 'emb', 'via', 'am', 'se', 'pm', 'ur', 'co', 'dz', 'riba', 'es', 'lei',
                           'gab', 'as', 'iai', 'aos', 'lid', 'da', 'csll', 'cf', 'vem', 'elan', 'nide', 'arma', 'if',
                           'toda', 'wis', 'fax', 'irl', 'cgpc', 'ktjr', 'ti', 'tr', 'up', 'vaz', 'fim', 'dito', 'loa',
                           'ata', 'para', 'ls', 'qual', 'pago', 'al', 'pb', 'ki', 'mp', 'nto', 'dia', 'esta', 'gest',
                           'ci', 'fig', 'pt', 'exas', 'jw', 'obn', 'pire', 'zs', 'out', 'ande', 'jk', 'vez', 'le', 'ne']

        if type(texto) is list:
            res = []
            for i, tr in enumerate(texto):
                UTIL.progress_bar(i,len(texto), 'analisando {}/{}'.format(i,len(texto)))
                res = res + UTIL_TEXTOS.possiveis_abreviacoes(tr, min_len)
                if i % 10 and len(res)>0:
                    print(set(res))
            return set(res)

        #texto = texto.replace('.', '. ')
        texto = re.sub('\s\s+', ' ', texto)
        termos = set(re.split(' ', texto.lower()))
        lista = []

        def _verifica(t):
            if 1 < len(t) <= min_len + 1 and t[-1] == '.':
                t = t.lower()[0:-1]
                if bool(re.match('^[a-z]+$', t)) \
                        and t not in NAO_ABREVIACOES \
                        and not ABREVIACOES_RGX.search(' '+t+'.'):
                    lista.append(t.lower())

        UTIL.map_thread(_verifica, termos, 10)
        return lista


if __name__ == "__main__":

    scores = UTIL_TEXTOS.sentencas('Esse texto tem 3 sentenças da Sr(a). e Dra. Maria e o Sr. João é uma. A outra com o art. 333 é a segunda. E por fim, temos fls. 5 a terceira.')
    [print(s) for s in scores]
    print('===============================================')

    print(ABREVIACOES_RGX.pattern)
    print(ABREVIACOES_RGX.search(' dra.'))
    print(ABREVIACOES_RGX.search(' 999.'))
    print('===============================================')

