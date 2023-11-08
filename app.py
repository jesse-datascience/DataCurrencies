# app.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

# Configuração de estilos e da pagina
sns.set_style("white")
palette = sns.color_palette("bright")
BG_COLOR = '#F0F0F0'

# Setup page
st.set_page_config(page_title="DataCurrencie Bank", page_icon="🤑", layout="wide")

# Função com cache para carregar o dataset
@st.cache_data(show_spinner=True)
def load_data():
    try:
        data = pd.read_feather('credit_scoring.ftr')
        return data
    except Exception as e:
        st.error(f"Erro ao carregar o dataset: {e}")
        return None

@st.cache_data(show_spinner=True)
def drop_duplicates(df):
    try:
        # Tenta remover duplicatas no DataFrame
        df_dropped = df.drop_duplicates()
        return df_dropped
    except AttributeError as e:
        # Captura o erro se 'df' não for um DataFrame pandas (por exemplo, se for uma lista ou None)
        st.error(f"Erro: O dado fornecido não é um DataFrame pandas válido. {e}")
        raise
    except KeyError as e:
        # Captura o erro se as colunas especificadas não existirem no DataFrame
        st.error(f"Erro: A chave especificada não foi encontrada no DataFrame. {e}")
        raise
    except Exception as e:
        # Captura qualquer outro erro não especificado anteriormente
        st.error(f"Ocorreu um erro inesperado: {e}")
        raise

# Função para carregar modelos
@st.cache_resource()
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Erro: Não foi possível encontrar o arquivo '{model_path}'. Verifique o caminho do arquivo.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None
    # Cache the model loading
@st.cache_resource()
def get_models():
    logistic_model = load_model('models/final_model.pkl')
    lgbm_model = load_model('models/final_model_lightgbm.pkl')
    return logistic_model, lgbm_model

def ColourWidgetText(wgt_txt, wch_colour = '#000000'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        elements[i].style.color = ' """ + wch_colour + """ '; } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)

# Função principal do Streamlit
def main():

    df = drop_duplicates(load_data())
    if df is None:
        st.write("Por favor, verifique se o arquivo 'credit_scoring.ftr' está no diretório correto ou se há algum outro problema com o arquivo.")
        return

    with st.chat_message("assistant"):
        st.markdown('### Olá, seja bem-vindo ao sistema interno do **DataCurrencie Bank**!')

    # Seção: Overview
    st.header('Data Overview')

    # Criando quatro colunas para exibir as métricas lado a lado
    col1, col2, col3, col4 = st.columns(4)

    # Número de clientes na primeira coluna
    num_clients = len(df)
    formatted_num_clients = f"{num_clients:,.0f}"
    col1.metric(label=":busts_in_silhouette: Número total de clientes", value=formatted_num_clients)

    # Renda anual média
    with col2:
        average_income = df['renda'].mean()
        col2.metric(label=":moneybag: Renda anual média", value=f"R${average_income:,.2f}")

    # Média de idade
    with col3:
        average_age = df['idade'].mean()
        col3.metric(label=":birthday: Idade média dos clientes", value=f"{average_age:.1f} anos")

    # Média de clientes que são devedores ('mau' == True)
    with col4:
        defaulters_ratio = df[df['mau']]['mau'].count() / num_clients
        col4.metric(label="📛 Clientes devedores", value=f"{defaulters_ratio:.2%}")

    # Aplicar as cores usando o método ColourWidgetText
    purple_color = '#800080'
    # Aplicar a cor roxa aos valores
    ColourWidgetText(formatted_num_clients, purple_color)
    ColourWidgetText(f"R${average_income:,.2f}", purple_color)
    ColourWidgetText(f"{average_age:.1f} anos", purple_color)
    ColourWidgetText(f"{defaulters_ratio:.2%}", purple_color)

    # Seção: Plots
    st.subheader('Data Plots')

    # Função para criar e exibir um gráfico de barras estilizado para uma variável
    def plot_categorical_variable(column, title):

        # Criando a figura com a cor de fundo especificada
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_COLOR)
        
        # Valores e etiquetas
        values = df[column].value_counts().values
        labels = df[column].value_counts().index
        # Traduzindo as etiquetas
        labels = ['Sim' if label == 'S' else 'Não' for label in labels]

        # Cores personalizadas para os dados
        colors = ['#9c88ff', '#ffe58a']  # roxo e amarelo claro

        # Plotando o gráfico de pizza no eixo definido
        ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))
        ax.set_title(title, fontsize=16)
        
        # Mudando a cor de fundo do eixo (área onde os dados são plotados)
        ax.set_facecolor(BG_COLOR)

        # Equal aspect ratio garante que o gráfico seja desenhado como um círculo.
        ax.axis('equal')

        st.pyplot(fig)

    def plot_age_distribution():
        # Cria a figura com o fundo personalizado
        plt.figure(figsize=(8, 6), facecolor=BG_COLOR)
        
        # Plota o histograma
        sns.histplot(data=df, x='idade', bins=30, color="#9c88ff", kde=True)
        
        # Títulos e etiquetas
        plt.title('')
        plt.xlabel('Idade', fontsize=14)
        plt.ylabel('Número de Clientes', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Remove as bordas do gráfico
        sns.despine(left=True, bottom=True)

        # Configurando a cor de fundo para os eixos
        ax = plt.gca()  # Pega o eixo atual
        ax.set_facecolor(BG_COLOR)  # Define a cor de fundo para o eixo
        
        # Mudando a cor de fundo da área de plotagem do gráfico
        plt.gcf().set_facecolor(BG_COLOR)
        
        # Exibe o gráfico no Streamlit
        st.pyplot(plt)

    # Criação das colunas
    col1, col2, col3 = st.columns(3)
    # Histograma da distribuição da idade
    with col1:
        st.subheader('👤Idade')
        try:
            plot_age_distribution()
        except Exception as e:
            st.write(f"❌ Erro ao gerar o gráfico da Distribuição da Idade: {e}")
    # Gráfico de 'posse_de_veiculo'
    with col2:
        st.subheader('🚗 Veículo')
        try:
            plot_categorical_variable('posse_de_veiculo', '')
        except Exception as e:
            st.write(f"❌ Erro ao gerar o gráfico de Posse de Veículo: {e}")
    # Gráfico de 'posse_de_imovel' 
    with col3:
        st.subheader('🏠 Imóvel')
        try:
            plot_categorical_variable('posse_de_imovel', '')
        except Exception as e:
            st.write(f"❌ Erro ao gerar o gráfico de Posse de Imóvel: {e}")
    
    def plot_employment_duration_distribution(df):
        
        # Cria a figura com o fundo personalizado
        plt.figure(figsize=(8, 5), facecolor=BG_COLOR)
        
        # Plota o histograma
        ax = sns.histplot(df['tempo_emprego'], bins=30, kde=True, color="#9c88ff")  # Usando roxo para o histograma
        
        # Títulos e etiquetas
        ax.set_title('')
        ax.set_xlabel('Anos de Emprego')
        ax.set_ylabel('Número de Clientes')
        
        # Remove as bordas do gráfico
        sns.despine(left=True, bottom=True)
        
        # Configura a cor de fundo para os eixos
        ax.set_facecolor(BG_COLOR)  # Define a cor de fundo para o eixo
        
        # Mudando a cor de fundo da área de plotagem do gráfico e da figura inteira
        plt.gcf().set_facecolor(BG_COLOR)
        
        # Exibe o gráfico no Streamlit
        st.pyplot(plt)

    def plot_income_distribution(df):
        # Aplicando a transformação logarítmica à renda
        df['renda_log'] = np.log(df['renda'] + 1)
        
        # Configuração da figura com a cor de fundo personalizada
        plt.figure(figsize=(8, 5), facecolor=BG_COLOR)
        
        # Plotagem do histograma
        ax = sns.histplot(df['renda_log'], bins=30, kde=True, color="#9c88ff")
        ax.set_title('Distribuição da Log da Renda dos Clientes')
        ax.set_xlabel('Log da Renda')
        ax.set_ylabel('Número de Clientes')
        
        # Remove as bordas do gráfico
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Configura a cor de fundo para os eixos
        ax.set_facecolor(BG_COLOR)
        
        # Mudando a cor de fundo da área de plotagem do gráfico e da figura inteira
        plt.gcf().set_facecolor(BG_COLOR)
        
        # Exibindo o gráfico no Streamlit
        st.pyplot(plt)

    # No corpo principal do Streamlit:
    col1, col2 = st.columns(2)  # Criando duas colunas

    with col1:
        st.subheader('Tempo de Trabalho 📊')
        plot_employment_duration_distribution(df)

    with col2:
        st.subheader('Log da Renda 💰')
        plot_income_distribution(df)
        st.write("A transformação logarítmica da renda nivela as distribuições de salários extremamente altos e baixos, facilitando a visualização e interpretação das tendências centrais e dispersões de renda na população.")

    # Secao: Outras Variaveis
    # Algumas estatísticas, Tem mais no .ipynb.
    st.subheader('Análise Bivariada')

    # Initialize a Streamlit app and set up the tab
    col1, col2 = st.columns(2)

    with col1:   
        # Crie a figura e um conjunto de subplots com a cor de fundo definida
        fig, axs = plt.subplots(figsize=(6, 4), facecolor=BG_COLOR)
        axs.set_facecolor(BG_COLOR)

        # Cria a tabela cruzada e a plota
        cross_tab = pd.crosstab(df['posse_de_veiculo'], df['posse_de_imovel'])
        cross_tab.plot(kind='bar', stacked=True, color=["purple", "yellow"], ax=axs)
        axs.set_title('')
        axs.set_xlabel('Posse de Veículo')
        axs.set_ylabel('Número de Clientes')
        axs.legend(title='Posse de Imóvel', labels=['Sim', 'Não'])

        # Remove as bordas
        for spine in axs.spines.values():
            spine.set_visible(False)

        # Exibe o gráfico na aplicação Streamlit
        st.pyplot(fig)

    with col2:    
        # Defina os limites e rótulos de idade
        limites = [20, 30, 40, 50, 60, 70]
        faixas_etarias = ["20-29", "30-39", "40-49", "50-59", "60+"]

        # Categoriza a 'idade' em faixas
        df['faixa_etaria'] = pd.cut(df['idade'], bins=limites, labels=faixas_etarias, right=False)
        
        # Calcula a média de filhos por grupo etário
        media_filhos_por_faixa = df.groupby('faixa_etaria')['qtd_filhos'].mean()
        
        # Cria a figura para o segundo gráfico com a cor de fundo definida
        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor=BG_COLOR)
        ax2.set_facecolor(BG_COLOR)

        # Plota a média de filhos por grupo etário
        media_filhos_por_faixa.plot(kind='bar', color='purple', alpha=0.7, ax=ax2)
        ax2.set_title('')
        ax2.set_xlabel('Faixa Etária')
        ax2.set_ylabel('Média de Filhos')

        # Remove as bordas
        for spine in ax2.spines.values():
            spine.set_visible(False)

        # Exibe o gráfico na aplicação Streamlit
        st.pyplot(fig2)

    # 2ª Seção: Upload e Análise da Base de Testes
    # Carrega os modelos
    logistic_model, lgbm_model = get_models()

    # 2ª Seção: Upload e Análise da Base de Testes
    st.header('Avaliação da Base de Clientes Recentes')

    # Descrição dos modelos
    st.markdown("""
        Antes de carregar sua base de clientes recentes para análise, selecione um dos modelos abaixo e leia mais sobre eles se desejar:

        - **Regressão Logística**: Um modelo clássico de classificação que pode ser mais robusto a outliers e menos propenso a sobreajuste. [Saiba mais](https://en.wikipedia.org/wiki/Logistic_regression).
        - **LightGBM**: Um modelo baseado em gradient boosting que é eficaz com grandes conjuntos de dados e capaz de lidar com um grande número de recursos. [Saiba mais](https://lightgbm.readthedocs.io/en/latest/).
        
        Ambos os modelos foram treinados com um conjunto de dados representativo.
    """)

    # Upload da base de dados de testes
    uploaded_file = st.file_uploader("Escolha um arquivo Feather", type=['ftr'])
    if uploaded_file is not None:
        # Attempt to read the uploaded file
        try:
            dataframe = pd.read_feather(uploaded_file)
            st.write("Pré-visualização dos dados carregados:", dataframe.head())

        except ValueError as ve:
            st.error('Formato de arquivo inválido. Por favor, forneça um arquivo Feather (.ftr) correto.')
            st.exception(ve)
            st.stop()
        except Exception as e:
            st.error('Erro desconhecido ao carregar o arquivo.')
            st.exception(e)
            st.stop()

        # Seleção do modelo
        selected_model = st.selectbox('Selecione o modelo para avaliação', ['Regressão Logística', 'LightGBM'])

        if st.button('Analisar'):
            with st.spinner('Analisando os dados...'):
                try:
                    # Realizando as previsões com o modelo selecionado
                    if selected_model == 'Regressão Logística':
                        predictions = logistic_model.predict(dataframe)
                    else:
                        predictions = lgbm_model.predict(dataframe)

                    # Mostrando os resultados de forma mais visual
                    st.subheader('Resultados da Previsão')
                    results_df = pd.DataFrame(predictions, columns=['Resultado'])
                    st.write(results_df)

                    # Disponibilizar um download com os resultados
                    st.download_button(
                        label="Download dos resultados",
                        data=results_df.to_csv(index=False).encode('utf-8'),
                        file_name='resultados_previsao.csv',
                        mime='text/csv',
                        key='download-csv'
                    )
                except Exception as e:
                    st.error('Erro durante a análise dos dados.')
                    st.exception(e)


if __name__ == "__main__":
    main()
