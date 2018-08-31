# comparador_elastic
Criando um comparador e um sumarizador de textos usando a base do elasticsearch como mapa de termos relevantes

## Comparando textos
Existem diversas formas de comparar um texto. Navegando pela web achei alguns algoritmos mas eu queria poder incluir similaridade textual, shingles (grupos de tokens) e comparar apenas termos mais relevantes dos textos. Percebi que grande parte do esforço para isso já é feito de forma muito eficiente pelo elasticsearch. Então segui os seguintes passos:

### 1. criar um índice no elasticsearch com um campo com stemmer removendo stopwords, um com shingles removendo stopwords e usando stemmer, e um com shingles apenas. Cada campo é um analisador diferente, permitindo uma comparação diferente.
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
         "filter":["standard","asciifolding", "lowercase","stop_br" , "stemmer_br","filtro_shingle"]
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
- O elastic permite simular um documento sem incluí-lo, retornando os termos que seriam retornados caso o documento existisse na base. Com isso, podemos usar a base como <i>corpus</i> de documentos. Podemos retornar os termos do documento (como um tokenizador usando o analyser do campo) e podemos também retornar os termos relevantes de um conjunto de documentos, bem como os seus pesos de acordo com os documentos do <i>corpus</i> da base (já calculados o TFIDF ou BM25). 
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
 similaridade = SIMILARIDADE.cosine_similarity(vector1, vector2)
```

# Disponibilizarei os códigos python em breve, bem como um exemplo de como usar esse mesmo algoritmo para sumarizar um texto encontrando as sentenças relevantes dele.
