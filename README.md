# NT PF

Atlas analitico sobre noticias de operacoes da Policia Federal brasileira. O projeto coleta a base publica de noticias, extrai o conteudo principal de cada pagina, organiza os dados em artefatos tabulares e abre um painel narrativo em Streamlit para explorar crimes, padroes temporais, clusters semanticos e series recorrentes.

## Fonte

Base publica consultada:

- [Noticias de operacoes da PF](https://www.gov.br/pf/pt-br/assuntos/noticias/noticias-operacoes?b_start:int=0)

## Objetivo

O foco do projeto e identificar quais tipos de crimes aparecem com maior recorrencia ao longo do tempo e observar correlacoes entre temas, contextos operacionais e distribuicao territorial. Um exemplo de pergunta que o trabalho tenta responder e como lavagem de dinheiro se relaciona com outros crimes em anos e contextos diferentes.

## O que o projeto faz

1. Coleta a listagem paginada de operacoes publicadas no portal da PF.
2. Estrutura os metadados basicos em CSV.
3. Abre cada noticia individualmente e extrai o conteudo principal em markdown.
4. Gera artefatos analiticos com classificacao, recorrencia, series semanticas, pares semelhantes e distribuicoes por ano.
5. Publica um painel em Streamlit para leitura exploratoria e narrativa dos resultados.

## Stack

- Python
- Pandas
- Streamlit
- Plotly
- scikit-learn
- BeautifulSoup
- requests
- docling

## Estrutura do repositorio

```text
NT_PF/
|-- data/
|   |-- analise_qualitativa/   # saidas geradas pelo pipeline
|   |-- noticias_markdown/     # markdown de cada noticia extraida
|   `-- reference/
|       `-- brazil_states.geojson
|-- scripts/
|   |-- pf_operacoes_pipeline.py
|   `-- pf_analise_qualitativa.py
|-- streamlit_app.py
|-- requirements.txt
`-- .gitignore
```

## Como executar

Crie o ambiente virtual e instale as dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Etapa 1: coletar o indice estruturado

```powershell
.\.venv\Scripts\python.exe .\scripts\pf_operacoes_pipeline.py collect --output-csv .\data\pf_operacoes_index.csv
```

### Etapa 2: extrair o conteudo das noticias

```powershell
.\.venv\Scripts\python.exe .\scripts\pf_operacoes_pipeline.py extract --index-csv .\data\pf_operacoes_index.csv --output-csv .\data\pf_operacoes_conteudos.csv --markdown-dir .\data\noticias_markdown --only-missing
```

### Etapa 3: gerar a analise qualitativa

```powershell
.\.venv\Scripts\python.exe .\scripts\pf_analise_qualitativa.py --output-dir .\data\analise_qualitativa
```

### Etapa 4: abrir o painel

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\streamlit_app.py
```

## Artefatos gerados

- `data/pf_operacoes_index.csv`: indice estruturado com titulo, subtitulo, data, tags e link.
- `data/pf_operacoes_conteudos.csv`: manifesto com o status da extracao de cada noticia.
- `data/noticias_markdown/*.md`: texto principal de cada noticia convertido para markdown.
- `data/analise_qualitativa/`: tabelas analiticas e relatorio narrativo.
- `streamlit_app.py`: painel para leitura exploratoria dos resultados.

## Partes do painel Streamlit

O `streamlit_app.py` organiza a leitura em uma navegacao lateral com seis partes principais. Cada uma responde a um tipo diferente de pergunta sobre o acervo.

### Panorama

E a porta de entrada do painel. Resume o corpus com metricas gerais, introduz a historia analitica do projeto e mostra uma visao ampla do conjunto de noticias antes do mergulho por tema. Serve para responder perguntas como volume total, distribuicao geral e dimensao do acervo analisado.

### Crimes e Modus

Explora os crimes rotulados e os modos de operacao identificados no corpus. Esta parte mostra recorrencia por ano, comparacoes de sinais ao longo do tempo e leituras territoriais por estado. E a secao para entender quais crimes aparecem mais, como eles evoluem e onde ganham maior intensidade.

### Clusters

Agrupa noticias semanticamente parecidas. Aqui o painel mostra o tamanho de cada cluster, termos dominantes, crimes mais frequentes, linha do tempo, mapa por estados citados e uma rede 3D de proximidade entre clusters. Nessa rede, a similaridade entre um cluster e outro e calculada a partir do corpus textual agregado de cada cluster, ou seja, pela uniao dos textos das noticias que pertencem a ele. Quando um cluster aparece solto, isso nao significa erro automaticamente: indica apenas que, no limiar atual da rede, o corpus agregado dele nao encontrou conexoes fortes o suficiente com os demais clusters. Esta secao ajuda a enxergar blocos tematicos do acervo, em vez de olhar noticia por noticia.

### Series Recorrentes

Mostra continuidades operacionais e cadeias de noticias muito proximas entre si ao longo do tempo. O painel separa uma visao executiva e uma exploracao detalhada, com ranking de series, filtros por tipo, intensidade e periodo. E a parte usada para identificar repeticao, persistencia e desdobramentos narrativos.

### Vizinhanca Semantica

Traz a leitura de caso. A partir de uma noticia-fonte, o painel recupera os vizinhos mais proximos por similaridade do cosseno, exibe o markdown extraido e mostra as noticias relacionadas. Serve para sair do agregado e voltar ao detalhe, inspecionando exemplos concretos de proximidade semantica.

### Artefatos

Funciona como inventario final do pipeline. Lista os principais arquivos produzidos pela analise e exibe o relatorio narrativo consolidado. Esta secao ajuda a conectar o painel visual com os artefatos tabulares e textuais gerados durante o processamento.

### Navegacao lateral

A barra lateral organiza o percurso sugerido de leitura do painel nesta ordem:

1. Panorama
2. Crimes e Modus
3. Clusters
4. Series Recorrentes
5. Vizinhanca Semantica
6. Artefatos

### Estado inicial sem dados

Quando os arquivos gerados pelo pipeline ainda nao existem, o app nao falha. Em vez disso, ele mostra uma tela de orientacao com os comandos necessarios para reconstruir os dados localmente antes de abrir o painel completo.

## Versao enxuta para GitHub

Este repositorio foi preparado para subir ao GitHub sem levar o volume inteiro de artefatos gerados localmente. Por isso, os seguintes caminhos ficam fora do versionamento:

- `data/pf_operacoes_index.csv`
- `data/pf_operacoes_conteudos.csv`
- `data/noticias_markdown/`
- `data/analise_qualitativa/*.csv`
- `data/analise_qualitativa/*.md`

O arquivo `data/reference/brazil_states.geojson` continua versionado porque e uma referencia estatica usada pelo mapa do painel.

Se voce clonar o projeto e abrir o app sem gerar os dados antes, o `streamlit_app.py` mostra uma tela de orientacao com os comandos necessarios para reconstruir os artefatos localmente.

## Observacoes

- O pipeline depende de acesso a rede para consultar o portal publico da PF.
- A primeira execucao pode demorar porque envolve raspagem, extracao textual e geracao de artefatos analiticos.
- O repositorio foi mantido enxuto de proposito para facilitar publicacao, clonagem e manutencao no GitHub.
