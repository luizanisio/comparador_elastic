# comparador_elastic
Criando um comparador e um sumarizador de textos usando a base do elasticsearch como mapa de termos relevantes

## Comparando textos
Existem diversas formas de comparar um texto. Navegando pela web achei alguns algoritmos interessantes, mas eu queria poder incluir similaridade textual, shingles (grupos de tokens) e comparar apenas termos mais relevantes dos textos atualizando facilmente o <b>corpus</b> de documentos. O <b>sklearn</b> permite fazer isso em poucas linhas (como no exemplo abaixo). Mas eu ainda queria um pouco mais. Queria a facilidade de manter um corpus atualizado dinamicamente e poder comparar textos usando os pesos desse corpus. Percebi que grande parte do esforço para isso já é feito de forma muito eficiente pelo elasticsearch. 

### Exemplo de comparação simples e eficaz usando o sklearn
- a documentação do TfidfVectorizer é bem clara, e pode ser acessada aqui: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- Esse link me ajudou muito no entendimento e uso do <b>TfidfVectorizer</b>, além da documentação: https://markhneedham.com/blog/2016/07/27/scitkit-learn-tfidf-and-cosine-similarity-for-computer-science-papers/ 
- mais abaixo tem uma matriz de similaridade feita com o sklearn e uma feita com o elastic

#### Com esse exemplo, pode-se comparar documentos facilmente. 

```py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
STOP_BR = stopwords.words('portuguese')

text1 = 'Teste do teste do teste'
text2 = 'Teste do teste do teste Teste do teste do teste Teste do teste do teste Teste do teste do teste da blá'
text3 = 'Esse teste não é nunhum blá blá blá'
textos=[text1,text2,text3]
corpus = ' '.join(textos)

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), strip_accents='ascii', stop_words=set(STOP_BR))
matriz_tfidf =  tf.fit_transform(textos,corpus)
cossenos = cosine_similarity(matriz_tfidf)
UTIL_MATRIZ.print_console('Cossenos:',cossenos)
```
Teremos algo assim como resposta:
```bat
     Cossenos:
     1.0                     0.9411973662740533      0.09892304334458238
     0.9411973662740533      1.0000000000000004      0.10227717325469488
     0.09892304334458238     0.10227717325469488     1.0
```

## Como eu quero complicar um pouco, mas não muito, vamos ao elastic
- Em resumo, vou buscar para cada documento os termos e pesos deles no corpus de documentos do elastic (de acordo com as regras dos analisadores criados, stemmer, stop words, sinônimos etc). Vou criar a matriz <b>csr_matrix</b> e calcular a similaridade pelo cosseno usando o <b>sklearn</b>.

### 1. criar um índice no elasticsearch com um campo com stemmer removendo stopwords, um com shingles removendo stopwords e usando stemmer, e um com shingles apenas. Cada campo é um analisador diferente, permitindo uma comparação diferente.

- segue arquivo stop_br.txt com alguns stops simples.
```json
PUT /comparador/
{    "analysis": {
     "filter": {
       "stop_br": {"type": "stop","stopwords_path": "stop_br.txt" },
       "stemmer_br": {"type": "stemmer", "language":   "brazilian" },
       "filtro_shingle":{ "type":"shingle", "max_shingle_size":3,
                          "min_shingle_size":2, "output_unigrams":"true"}
     },
     "analyzer": {
       "texto_br": {
         "tokenizer":  "standard",
         "filter": ["lowercase","asciifolding","stop_br","stemmer_br"]
       },
       "shingle_br":{
         "tokenizer":"standard",
         "filter":["standard","asciifolding", "lowercase" , "stop_br", "stemmer_br","filtro_shingle"]
       },
       "shingle_raw":{
         "tokenizer":"standard",
         "filter":["standard","asciifolding", "lowercase", "filtro_shingle"]
       }
      }
   }
}

PUT /comparador/_mapping/textos/
{ "properties": {
       "Id": { "type": "keyword"},
       "Texto": { "type": "text", "analyzer" : "texto_br",
                                "term_vector": "with_positions_offsets"  },
       "Texto_Shingle": { "analyzer": "shingle_br", "type":"text" },
       "Texto_Shingle_RAW": { "analyzer": "shingle_raw", "type":"text" },
       "Atualizacao": { "type": "date", 
          "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd HH:mm:ss.SSS||yyyy-MM-dd" }
     }
}
```

### 2. incluir no elastic diversos textos do domínio da área que se deseja avaliar os documentos. Nesse caso vou usar textos jurídicos, os mesmos textos usados para gerar o vetor de palavras do exemplo word2vec.
- A inclusão pode ser feita via código ou kibana, vou disponibilizar um código que inclui todos os textos de uma pasta.
´´´json
  POST comparador/textos/1
  { "Id" : 1,
    "Texto" : "1. Adeus despedida de encomendar [a-deus] 2. Alegoria representa uma coisa para dar ideia de outra 3. Amanhecer nascer do sol, presenciável em noites de trabalho ou de insónia ",
    "Texto_Shingle" : "1. Adeus despedida de encomendar [a-deus] 2. Alegoria representa uma coisa para dar ideia de outra 3. Amanhecer nascer do sol, presenciável em noites de trabalho ou de insónia ",
    "Texto_Shingle_Raw" : "1. Adeus despedida de encomendar [a-deus] 2. Alegoria representa uma coisa para dar ideia de outra 3. Amanhecer nascer do sol, presenciável em noites de trabalho ou de insónia ",
    "Atualizacao" : "2018-01-01 10:22:33"
  }
´´´

### 3. fazer uma requisição ao elastic solicitando que retorne os termos importantes de um documento de acordo com o analyzer de um campo.
- O elastic permite simular um documento sem incluí-lo, retornando os termos/tokens que seriam retornados caso o documento existisse na base. Com isso, podemos usar a base como <i>corpus</i> de documentos. Podemos retornar os termos do documento (como um tokenizador usando o analyser do campo) e podemos também retornar os termos relevantes de um conjunto de documentos, bem como os seus pesos de acordo com os documentos do <i>corpus</i> da base (já calculados o TFIDF ou BM25). 
- dos tokens, geramos um contador de tokens (quantas vezes cada token aparece em cada documento)
- dos pesos, geramos um multiplicador de pesos e multiplicamos o peso de cada termos pelo contador de termos de cada documento
- daí temos o peso de cada termo em cada documento de acordo com o peso de cada termo na base, algo como {"casa" : 12.76355, "carro" : 13.7665} ...

- <b>buscando o peso dos termos dos dois documentos</b>
```json
POST comparador/textos/_termvectors
{"doc": {
  "Texto_Shingle" : "Resolvi consolidar alguns resultados de estudos realizados com o uso de elasticsearch e python para facilitar o trabalho de quem está iniciando nessa área. Esse não é um trabalho acadêmico e não visa esgotar todo o assunto. Alguns tópicos abordados aqui com exemplos funcionais e dicas de como evoluí-los e usá-los no dia-a-dia. Estão mais próximos de receitas do tipo pegar, adaptar e usar. O outro documento está aqui também ... "},
                 "field_statistics": false,
                 "term_statistics": true,
                 "positions": false,
                 "offsets": false,
                 "filter": {}}
```

O retorno do exemplo acima, com apenas uma dezena de documentos na base, seria próximo disso (só listei alguns termos):
```json
  {
   "_index": "comparador",
   "_type": "textos",
   "_version": 0,
   "found": true,
   "took": 7,
   "term_vectors": {
   "Texto_Shingle": {
    "terms": { "acord com os": { "term_freq": 1, "score": 1.9162908 },
    "aqu vai": { "term_freq": 1, "score": 1.9162908 },
    "aqu vai um": { "term_freq": 1, "score": 1.9162908 },
    "com": { "term_freq": 2, "score": 3.8325815 },
    "com os": { "term_freq": 1, "score": 1.9162908 },
    "com os term": { "term_freq": 1, "score": 1.9162908 },
    "com seus": { "term_freq": 1, "score": 1.9162908 },
    "com seus pes": { "term_freq": 1, "score": 1.9162908 },
    "de": { "doc_freq": 1, "ttf": 1, "term_freq": 2, "score": 3.8325815  },
    "de acord": { "term_freq": 1, "score": 1.9162908  }
    }  }
  } }
```
- <b>buscando os tokens de cada documento</b>
```json
POST comparador/_analyze
{"field": "Texto_Shingle", "text": "Resolvi consolidar alguns resultados de estudos realizados com o uso de elasticsearch e python para facilitar o trabalho de quem está iniciando nessa área. Esse não é um trabalho acadêmico e não visa esgotar todo o assunto. Alguns tópicos abordados aqui com exemplos funcionais e dicas de como evoluí-los e usá-los no dia-a-dia. Estão mais próximos de receitas do tipo pegar, adaptar e usar."}
```

O retorno do exemplo acima será a lista de tokens do documento, de acordo com o analyzer do campo (abaixo somente alguns dos tokens retornados):
```json
{ "tokens": [
 { "token": "resolv", "start_offset": 0, "end_offset": 7, "type": "<alphanum>", "position": 0 }, 
 { "token": "resolv consolid", "start_offset": 0, "end_offset": 18, "type": "shingle", "position": 0, "positionlength": 2 },
 { "token": "resolv consolid alguns", "start_offset": 0, "end_offset": 25, "type": "shingle", "position": 0, "positionlength": 3},
 { "token": "consolid", "start_offset": 8, "end_offset": 18, "type": "<alphanum>", "position": 1 }, 
 { "token": "consolid alguns", "start_offset": 8, "end_offset": 25, "type": "shingle", "position": 1, "positionlength": 2 },
 { "token": "consolid alguns result", "start_offset": 8, "end_offset": 36, "type": "shingle", "position": 1, "positionlength": 3 },
 { "token": "alguns", "start_offset": 19, "end_offset": 25, "type": "<alphanum>", "position": 2 } ]
}
```

### 4. calcular a distância do cosseno dos termos mais relevantes e seus pesos para os dois documentos. Daí temos um cálculo rápido e razoável de similaridade.
- com os vetores de termos e pesos de cada documento, basta calcular a distância entre os documentos 
```py
 cossenos = cosine_similarity(matriz_csr)
```

- Usando alguns textos de exemplo, podemos ter a matriz de similaridade entre os documentos conforme o exemplo abaixo.
![Exemplo de matriz com o uso do elastic](/imagens/comp_elastic.png)

- A mesma lista de documentos comparados usando o código do sklearn no ínicio do texto.
![Exemplo de matriz com o uso do sklearn](/imagens/comp_sklearn.png)

- Nos exemplos eu incluí um texto em inglês, para que fosse totalmente diferente de qualquer outro texto (<b>z_controle_en.txt</b>), e um texto em português com assunto diferente dos outros textos (<b>z_controle_pt.txt</b>).

### 5. O código de exemplo.
- Alimentando o índice do elastic com todos os textos de uma pasta. Crie antes os analisadores e o índice conforme descrito acima.

```py
    elastic = ELASTIC('http://localhost:9200', 'comparador', 'textos')
    alimentar_comparador(pasta='.\\textos_corpus\\',objElastic=elastic)
```

- Comparando todos os textos de uma pasta usando a classe exemplo do elastic.

```py
    cos_es = UTIL_SIMILARIDADE.documentos_matriz_elastic(pasta=pasta, campo_elastic='Texto_Shingle_RAW', objElastic=elastic)
    if cos_es is not None:
       UTIL_MATRIZ.gravar(arquivo='similaridade_es.txt', matriz=cos_es)
```

- Comparando todos os textos de uma pasta usando a classe exemplo do sklearn.

```py
    cos_sk = UTIL_SIMILARIDADE.documentos_matriz(pasta=pasta)
    if cos_sk is not None:
       UTIL_MATRIZ.gravar(arquivo='similaridade_sk.txt', matriz=cos_sk)
```

- A classe ELASTIC é simples e permite fazer o CRUD e usos diversos do elasticsearch.
- A classe UTIL_SIMILARIDADE contém métodos de comparação entre documentos de uma pasta e entre dois documentos.
- Outras classes acessórias estão disponíveis para que o projeto funcione.
- Pode-se obter a similaridade entre dois documentos com ou sem o elastic conforme o exemplo abaixo (disponível no arquivo "classes_similaridade.py"):

```py
    print('\nComparando dois textos com o sklearn: ', UTIL_SIMILARIDADE.compara(texto1=parecido, texto2=original))

    print('\nComparando dois textos com o elastic: ',
          UTIL_SIMILARIDADE.compara_elastic(texto1=parecido, texto2=original,
                                            campo_elastic='Texto_Shingle', objElastic=elastic))
```

## Resumindo textos: resumo por extração
Encontrei vários códigos e exemplos de como resumir um texto. Verifiquei que o mais complicado não é a técnica de resumo, mas a divisão do texto em sentenças para que estas possam ser <i>rankeadas</i> e com isso possamos selecionar as mais importantes para o documento. Em relação ao score das sentenças, o trabalho já foi feito na parte de comparação. Calculando os termos mais relevantes de cada sentença como se fossem documentos, onde o documento é o <b>corpus</b>. Como já temos duas formas de calcular o score de um documento, e agora podemos usar para calcular o score das sentenças, também temos duas formas de resumir. Uma usando o sklearn para identificar o score das sentenças, e outra usando o elasticsearch. 

- tipos de resumos automáticos: https://en.wikipedia.org/wiki/Automatic_summarization

### Exemplo de como criar um resumo usando o sklearn e o elasticsearch

- Calculando os scores: o código de exemplo usa a matriz csr para calcular o score de cada sentença, somando o peso dos termos para cada uma e ordenando as sentenças de forma decrescente pelos seus scores.
- Com isso é só decidir quantas sentenças queremos retornar, e retorná-las na ordem natural do texto. No exemplo temos a opção de definir um número mínimo de sentenças e um número mínimo do percentual de scores do texto. Quando os dois mínimos forem atingidos, o resumo é concluído.
- O arquivo "resumir.txt" é uma cópia do texto sobre Platão da wikipedia, e é usado para o resumo exemplo, que é feito buscando no mínimo 1% do texto, com no mínimo 2 sentenças.
```py
    texto = UTIL_ARQUIVOS.carregar_string_arquivo('.\\textos_corpus\\resumir.txt')

    print('== RESUMO SKLEARN')
    print(UTIL_SIMILARIDADE.resumo_textos(texto_ou_textos=texto, min_sentencas=2, min_percentual=1))

    print('== RESUMO ELASTIC')
    print(UTIL_SIMILARIDADE.resumo_textos(texto_ou_textos=texto, min_sentencas=2, min_percentual=1,
                                          objElastic=elastic, campo_elastic='Texto_Shingle'))
```
- Os resultados estão abaixo. Interessante que nesse caso os dois resumos ficaram iguais. O que vai alterar os pesos de cada um é o peso dos termos colocados no elasticsearch, já que o sklearn está usando como corpus apenas o próprio documento. O valor dos scores não é importante, o elastic e o sklearn calculam de forma diferente. No algoritmo usado, foi incluído um peso extra para sentenças completamente em maiúsculo, mas isso pode ser alterado no parâmetro da chamada do método <b>resumo_textos</b>.
```bat
== RESUMO SKLEARN
Percentual:  1.9664313580305095 Sentenças:  2 Total scores:  1127.0736074334588
A mais famosa fonte da história do resgate de Platão por Arquitas está na Sétima Carta, onde Platão descreve seu envolvimento nos incidentes de seu amigo Dion de Siracusa e Dionísio I, o tirano de Siracusa, Platão esperava influenciar o tirano sobre o ideal do rei-filósofo (exposto em Górgias, anterior à sua viagem), mas logo entrou em conflito com o tirano e sua corte; mas mesmo assim cultivou grande amizade com Díon, parente do tirano, a quem pensou que este pudesse ser um discípulo capaz de se tornar um rei-filósofo.  Diógenes Laércio conta que ele "foi a Cirene, juntar-se a Teodoro, o matemático, depois à Itália, com os pitagóricos Filolau e Eurito; e daí para o Egito, avistar-se com os profetas; ele tinha decidido encontrar-se também com os magos, mas a guerras da Ásia o fizeram renunciar a isso" Apesar desse relato de Diógenes Laércio, é posto em dúvida se Platão foi mesmo ao Egito, pois há evidências de que a estadia foi inventada no Egito, para aproximar Platão à tradição de sabedoria egípcia.
== RESUMO ELASTIC
Percentual:  3.904509364856055 Sentenças:  2 Total scores:  11529.746356458076
A mais famosa fonte da história do resgate de Platão por Arquitas está na Sétima Carta, onde Platão descreve seu envolvimento nos incidentes de seu amigo Dion de Siracusa e Dionísio I, o tirano de Siracusa, Platão esperava influenciar o tirano sobre o ideal do rei-filósofo (exposto em Górgias, anterior à sua viagem), mas logo entrou em conflito com o tirano e sua corte; mas mesmo assim cultivou grande amizade com Díon, parente do tirano, a quem pensou que este pudesse ser um discípulo capaz de se tornar um rei-filósofo.  Diógenes Laércio conta que ele "foi a Cirene, juntar-se a Teodoro, o matemático, depois à Itália, com os pitagóricos Filolau e Eurito; e daí para o Egito, avistar-se com os profetas; ele tinha decidido encontrar-se também com os magos, mas a guerras da Ásia o fizeram renunciar a isso" Apesar desse relato de Diógenes Laércio, é posto em dúvida se Platão foi mesmo ao Egito, pois há evidências de que a estadia foi inventada no Egito, para aproximar Platão à tradição de sabedoria egípcia.
```
- Pode-se também listar o score das sentenças, onde cada item do array corresponde à (posição da sentença no texto, score, texto):
```py
    print('== SCORES SKLEARN')
    [print(s) for s in UTIL_SIMILARIDADE.scores_textos(texto_ou_textos=texto, peso_so_maiusculas=2)]
```
- Como resultado temos (coloquei apenas as primeiras linhas):
```bat
(50, 11.445386362363422, 'A mais famosa fonte da história do resgate de Platão por Arquitas está na Sétima Carta, onde Platão descreve seu envolvimento nos incidentes de seu amigo Dion de Siracusa e Dionísio I, o tirano de Siracusa, Platão esperava influenciar o tirano sobre o ideal do rei-filósofo (exposto em Górgias, anterior à sua viagem), mas logo entrou em conflito com o tirano e sua corte; mas mesmo assim cultivou grande amizade com Díon, parente do tirano, a quem pensou que este pudesse ser um discípulo capaz de se tornar um rei-filósofo. ')
(46, 10.717742482293794, 'Diógenes Laércio conta que ele "foi a Cirene, juntar-se a Teodoro, o matemático, depois à Itália, com os pitagóricos Filolau e Eurito; e daí para o Egito, avistar-se com os profetas; ele tinha decidido encontrar-se também com os magos, mas a guerras da Ásia o fizeram renunciar a isso" Apesar desse relato de Diógenes Laércio, é posto em dúvida se Platão foi mesmo ao Egito, pois há evidências de que a estadia foi inventada no Egito, para aproximar Platão à tradição de sabedoria egípcia. ')
(228, 10.022613240910918, 'Já para o filólogo alemão Ulrich von Wilamowitz-Moellendorff, Platão teria nascido quando Diótimos era arconte epônimo, mais especificamente entre 29 de julho de 428 a. C. e 24 de julho de 427 a. C. O filólogo grego acredita que o filósofo teria nascido em 26 ou 27 de maio de 427 a. C. , enquanto o filósofo britânico Jonathan Barnes estipula 428 a. C. como o ano de nascimento de Platão. ')
(94, 9.843437827272652, 'Segundo Diógenes Laércio(III, 61), se encontravam na nona tetralogia "uma carta a Aristodemo [de fato a Aristodoro]" (X), duas a Arquitas (IX, XII), quatro a Dionísio II (I, II, III, IV), uma a Hérmias, Erastos e Coriscos (VI), uma a Leodamas (XI), uma a Dion (IV), uma a Perdicas (V) e duas aos parentes de Dion (VII, VIII)". ')
(42, 9.74490045086472, 'Mas, a situação política após a restauração da democracia ateniense em 403 também o desagradou, sendo um ponto de viragem na vida de Platão, a execução de Sócrates em 399 a. C, que o abalou profundamente, levando-o a avaliiar a ação do Estado contra seu professor, como uma expressão de depravação moral e evidência de um defeito fundamental no sistema político. ')
(200, 9.70703193005209, 'Acredita-se que Pletão passou uma cópia dos diálogos platônicos para Cosme de Médici em 1438/39 durante o Conselho de Ferrara, quando foi chamado para unificar as Igrejas grega e latina e então foi transferido para Florença onde fez uma palestra sobre a relação e as diferenças de Platão e Aristóteles; assim, Pletão teria influenciado Cosme com seu entusiasmo. ')
(137, 9.41945973331433, '. . Platão foi certamente o representante máximo desse gênero literário, superior a todos os outros e, mesmo, o único representante, pois apenas em seus escritos é que se pode reconhecer a natureza autêntica do filosofar socrático, que nos outros escritores, degenerou em maneirismos; sendo assim, o diálogo, em Platão, é mais do que um gênero literário: é sua forma de fazer filosofia. ')
```
