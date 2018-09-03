from nltk import download as nldw
from nltk.corpus import stopwords
#nldw('stopwords')
STOP_BR = stopwords.words('portuguese')

from requests.auth import HTTPBasicAuth
import requests
import json
from collections import Counter
import time

from classes_util import UTIL_COUNTER

class ELASTIC:
    def __init__(self, host='http://elasticsearch:9200', indice='', tipo='', usuario=None, senha=None):
        self.host = host
        self.indice = indice
        self.tipo = tipo
        self.usr = usuario
        self.psd = senha

    def url_elastic(self, indice=True, tipo=True, funcionalidade=None):
        result = self.host
        if indice: result = result + '/' + self.indice
        if tipo: result = result + '/' + self.tipo
        if funcionalidade: result = result + '/' + funcionalidade
        return result

    def url_search(self):
        return self.url_elastic(True, True, '_search')

    def url_analyze(self):
        return self.url_elastic(True, False, '_analyze')

    def url_term_vectors(self):
        return self.url_elastic(True, True, '_termvectors')

    def url_indice(self):
        return self.url_elastic(True, False)

    def url_tipo(self):
        return self.url_elastic(True, True)

    def requisicao(self, url, json_data, metodo):
        if metodo == 'GET':
            r = requests.get(url, json=json_data, auth=HTTPBasicAuth(self.usr, self.psd))
        elif metodo == 'POST':
            r = requests.post(url, json=json_data, auth=HTTPBasicAuth(self.usr, self.psd))
        elif metodo == 'PUT':
            r = requests.put(url, json=json_data, auth=HTTPBasicAuth(self.usr, self.psd))
        elif metodo == 'DELETE':
            r = requests.delete(url, auth=HTTPBasicAuth(self.usr, self.psd))
        return json.loads(r.content.decode())

    def get(self, url, json_data=None):
        resultado = self.requisicao(url, json_data, 'GET')
        return resultado

    def post(self, url, json_data):
        resultado = self.requisicao(url, json_data, 'POST')
        return resultado

    def put(self, url, json_data):
        resultado = self.requisicao(url, json_data, 'PUT')
        return resultado

    def delete(self, url, json_data):
        resultado = self.requisicao(url, json_data, 'DELETE')
        return resultado

    def search(self, json_data=None):
        if json_data:
            return self.post(self.url_search(), json_data)
        else:
            return self.get(self.url_search())

    def post_doc(self, json_data, id=None):
        url=self.url_elastic(indice=True, tipo=True, funcionalidade=id)
        resultado = self.requisicao(url, json_data, 'POST')
        return resultado

    def analyze(self, analyzer='', campo='', texto=''):
        if campo and (campo != ''):
            dados = {"field": campo, "text": texto}
        else:
            dados = {"analyzer": analyzer, "text": texto}
        resultado = self.post(self.url_analyze(), dados)
        if resultado.get('status') == 404:
            raise ValueError(''.join(['Não foi possível identificar o analyzer especificado [', analyzer,
                                      '] no índice [', self.indice, '] - erro 404']))
        if resultado:
            return resultado['tokens']
        else:
            return {}

    def analyze_counter(self, analyzer='', campo='', texto='', retirar_stop=False):
        an = self.analyze(analyzer, campo, texto)
        if not an:
            return None
        return Counter([k.get('token') for k in an if len(k.get('token')) > 2 and
                        ( (not retirar_stop) or k.get('token') not in STOP_BR )])

    def term_vector(self, campo, texto):
        dados = {"doc": {campo: texto},
                 "field_statistics": False,
                 "term_statistics": True,
                 "positions": False,
                 "offsets": False,
                 "filter": {}}
        termos = self.post(self.url_term_vectors(), dados)
        if termos:
            if termos.get('status') == 404:
                raise ValueError(''.join(['Não foi possível identificar os vetores do campo especificado [', campo,
                                          '] não é inválido para o índice [', self.indice, '/', self.tipo,
                                          '] - erro 404']))
            try:
                #print(termos)
                termos = termos.get('term_vectors')
                if termos:
                    termos = termos.get(campo)['terms']
            except KeyError:
                termos = {}
        else:
            termos = {}
        return termos

    def term_vector_counter(self, campo, texto, normalizado=False):
        termos = self.term_vector(campo, texto)
        if not termos:
            return Counter({})
        try:
            pesos = {k: termos.get(k).get('score') for k in termos.keys()}
        except KeyError:
            pesos = Counter({})
        if normalizado:
            return UTIL_COUNTER.normaliza_01(pesos)
        else:
            return pesos

    def datahora(self):
        return time.strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    elastic = ELASTIC('http://localhost:9200', 'comparador', 'textos')

    print('>>>>>>> Teste do analyze')
    print('URL: ' + elastic.url_analyze())
    print('Termos: ',elastic.analyze_counter('', 'Texto_Shingle', 'esse é o meu texto de teste dos textos testados como teste'))

    print('>>>>>>> Teste do term vector')
    print('URL: ' + elastic.url_term_vectors())
    pesos = elastic.term_vector_counter('Texto_Shingle', 'esse é o meu texto de teste dos textos testados como teste')
    print('Pesos: ', UTIL_COUNTER.ordena(pesos, False))

    print('>>>>>>> Teste do analyze com os pesos')
    print('URL: ' + elastic.url_analyze())
    an = elastic.analyze_counter('', 'Texto_Shingle', 'esse é o meu texto de teste dos textos testados como teste')
    an = UTIL_COUNTER.ajusta_pesos(counter_origem=an, counter_pesos=pesos)
    print('Termos e pesos: ', UTIL_COUNTER.ordena(an, True))

    print('\Testes concluídos!')

