# DataCurrencies

Bem-vindo ao projeto **DataCurrencies**.

## Sumário

- [Etapas do Projeto](#etapas-do-projeto)
- [Começando](#começando)
- [Contribuição](#contribuição)
- [Licença](#licença)
- [Contato](#contato)

## Etapas do Projeto

O projeto é estruturado em três etapas fundamentais:

### 1. ETL de Dados Brutos

O processo começa com a **Extração, Transformação e Carregamento (ETL)** dos dados brutos. Nesta etapa, lidamos com a aquisição dos dados, limpando-os e transformando-os em um formato estruturado e pronto para análise.

### 2. Análise Exploratória de Dados (EDA)

Uma vez que os dados estejam limpos e prontos, mergulhamos neles através da **Análise Exploratória de Dados (EDA)**. Esta etapa nos permite compreender as características intrínsecas dos dados, identificar padrões, anomalias e relações fundamentais entre as variáveis.

### 3. Modelagem em uma Regressão Logística

Com um entendimento dos dados em mãos, passamos para a etapa de **Modelagem**. Utilizamos a regressão logística para entender e quantificar as relações entre nossas variáveis independentes e dependentes. Este modelo nos proporciona uma maneira robusta de prever e inferir fenômenos a partir dos dados.

## App Streamlit

Este documento fornece instruções sobre como utilizar o script app.py, que é um aplicativo de Streamlit construído para visualizar e prever dados de pontuação de crédito.

### Pré-requisitos

Para rodar o aplicativo, você precisará ter os seguintes itens instalados:

- Python 3.6 ou superior
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- LightGBM
- Pickle

### Rodando o Aplicativo
Para iniciar o aplicativo, navegue até o diretório contendo app.py e execute o seguinte comando no terminal: streamlit run app.py

### Funcionalidades do Aplicativo

1. **Carregar e Visualizar Dados**
   - O aplicativo inicia carregando um conjunto de dados de pontuação de crédito (espera-se um arquivo chamado `credit_scoring.ftr`).
   - Métricas chave são exibidas, como número total de clientes, renda anual média, idade média dos clientes e percentual de clientes devedores.
   - São gerados gráficos para distribuições de idade, posse de veículo, posse de imóvel, etc.

2. **Análise Bivariada**
   - São exibidos gráficos que mostram relações entre diferentes variáveis, como posse de veículo e posse de imóvel, bem como número médio de filhos por faixa etária.

3. **Avaliação da Base de Clientes Recentes**
   - É possível fazer upload de um arquivo CSV para a base de clientes recentes e selecionar um modelo de predição.
   - O modelo escolhido (Regressão Logística ou LightGBM) é utilizado para fazer previsões na nova base de clientes.
   - Os resultados da previsão podem ser baixados como um arquivo CSV.


## Contribuição

Sua contribuição é muito bem-vinda!

## Licença

Este projeto está licenciado sob a Licença MIT.
## Contato

Caso tenha dúvidas, sugestões ou feedbacks, fique à vontade para nos contatar.

---

