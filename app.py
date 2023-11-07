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

# Configuração de estilos do Seaborn
sns.set_style("white")
palette = sns.color_palette("bright")

# Função com cache para carregar o dataset
@st.cache_data(show_spinner=False)
def load_data():
    try:
        data = pd.read_feather('credit_scoring.ftr')
        return data
    except Exception as e:
        st.error(f"Erro ao carregar o dataset: {e}")
        return None
    
# Função para carregar modelos
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

def ColourWidgetText(wgt_txt, wch_colour = '#000000'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        elements[i].style.color = ' """ + wch_colour + """ '; } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)

# Função principal do Streamlit
def main():

    df = load_data()
    if df is None:
        st.write("Por favor, verifique se o arquivo 'credit_scoring.ftr' está no diretório correto ou se há algum outro problema com o arquivo.")
        return

    with st.chat_message("assistant"):
        st.write('Olá, seja bem vindo ao sistema interno do DataCurrencie Bank')

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
        defaulters_ratio = df[df['mau'] == True]['mau'].count() / num_clients
        col4.metric(label=":warning: % de clientes devedores", value=f"{defaulters_ratio:.2%}")

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
        plt.figure(figsize=(8, 6))
        
        # Valores e etiquetas
        values = df[column].value_counts().values
        labels = df[column].value_counts().index
        # Traduzindo as etiquetas
        labels = ['Sim' if label == 'S' else 'Não' for label in labels]

        # Cores personalizadas
        colors = ['#9c88ff', '#ffe58a']  # roxo e amarelo claro

        # Plotando o gráfico de pizza
        plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))
        plt.title(title, fontsize=16)
        plt.axis('equal')  # Equal aspect ratio garante que o gráfico seja desenhado como um círculo.
        
        st.pyplot(plt)

    def plot_age_distribution():
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x='idade', bins=30, color="#9c88ff", kde=True)  # roxo
        plt.title('', fontsize=16)
        plt.xlabel('Idade', fontsize=14)
        plt.ylabel('Número de Clientes', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        sns.despine(left=True, bottom=True)  # Remove bordas do gráfico
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
        plt.figure(figsize=(8, 5))
        ax = sns.histplot(df['tempo_emprego'], bins=30, kde=True, color="#9c88ff")  # Usando roxo para o histograma
        ax.set_title('')
        ax.set_xlabel('Anos de Emprego')
        ax.set_ylabel('Número de Clientes')
        
        # Remove the spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Show the plot in the Streamlit app
        st.pyplot(plt)

    def plot_income_distribution(df):
        # Aplicando a transformação logarítmica à renda
        df['renda_log'] = np.log(df['renda'] + 1)
        
        # Set up the figure
        plt.figure(figsize=(8, 5))
        ax = sns.histplot(df['renda_log'], bins=30, kde=True, color="#9c88ff")
        ax.set_title('')
        ax.set_xlabel('Log da Renda')
        ax.set_ylabel('Número de Clientes')
        
        # Remove the spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Show the plot in the Streamlit app
        st.pyplot(plt)

    # No corpo principal do Streamlit:
    col1, col2 = st.columns(2)  # Criando duas colunas

    with col1:
        st.subheader('Tempo de Trabalho 📊')
        plot_employment_duration_distribution(df)

    with col2:
        st.subheader('Log da Renda 💰')
        plot_income_distribution(df)
        st.write("A transformação logarítmica da renda, conhecida como 'log renda', nivelada as distribuições de salários extremamente altos e baixos, facilitando a visualização e interpretação das tendências centrais e dispersões de renda na população.")
    # Você pode adicionar detalhes sobre a quantidade de dados, distribuições, estatísticas descritivas, etc.

    # Definição de uma função para criar um histograma com base em uma coluna
    def plot_histogram(column_name, title):
        plt.figure(figsize=(8, 6))
        df[column_name].value_counts().plot(kind='bar')
        plt.title(title)
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Secao: Outras Variaveis
    # Algumas estatísticas, Tem mais no .ipynb.
    st.subheader('Análise Bivariada')

    # Initialize a Streamlit app and set up the tab
    col1, col2 = st.columns(2)

    with col1:        
        # Create a figure and a set of subplots
        fig, axs = plt.subplots()

        # Create the crosstab and plot it
        cross_tab = pd.crosstab(df['posse_de_veiculo'], df['posse_de_imovel'])
        cross_tab.plot(kind='bar', stacked=True, color=["purple", "yellow"], ax=axs)
        axs.set_title('')
        axs.set_xlabel('Posse de Veículo')
        axs.set_ylabel('Número de Clientes')
        axs.legend(title='Posse de Imóvel', labels=['Sim', 'Não'])

        # Remove borders
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.spines['bottom'].set_visible(False)
        axs.spines['left'].set_visible(False)

        # Show the plot in the Streamlit app
        st.pyplot(fig)

    with col2:    
        # Define the age bins and labels
        limites = [20, 30, 40, 50, 60, 70]
        faixas_etarias = ["20-29", "30-39", "40-49", "50-59", "60+"]
        
        # Categorize the 'idade' into bins
        df['faixa_etaria'] = pd.cut(df['idade'], bins=limites, labels=faixas_etarias, right=False)
        
        # Calculate the mean children per age group
        media_filhos_por_faixa = df.groupby('faixa_etaria')['qtd_filhos'].mean()
        
        # Create a figure for the second plot
        fig2, ax2 = plt.subplots()
        
        # Plot the average number of children by age group
        media_filhos_por_faixa.plot(kind='bar', color='purple', alpha=0.7, ax=ax2)
        ax2.set_title('')
        ax2.set_xlabel('Faixa Etária')
        ax2.set_ylabel('Média de Filhos')

        # Remove borders
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        st.pyplot(fig2)

    # 2ª Seção: Upload e Análise da Base de Testes
    st.header('2. Avaliação da Base de Clientes Recentes')
    
    # Carregando os modelos (substitua com o caminho correto para seus modelos)
    logistic_model = load_model('models/final_model.pkl')
    lgbm_model = load_model('models/final_model_lightgbm.pkl')

    # Upload da base de dados de testes
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=['csv'])
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)

        # Seleção do modelo
        selected_model = st.selectbox('Selecione o modelo para avaliação', ['Regressão Logística', 'LightGBM'])

        if st.button('Analisar'):
            try:
                # Realizando as previsões com o modelo selecionado
                if selected_model == 'Regressão Logística':
                    predictions = logistic_model.predict(dataframe)
                else:
                    predictions = lgbm_model.predict(dataframe)

                # Mostrando os resultados
                st.subheader('Resultados da Previsão')
                st.write(predictions)

                # Disponibilizar um download com os resultados
                result_df = pd.DataFrame(predictions, columns=['Resultado'])
                st.download_button(
                    label="Download dos resultados",
                    data=result_df.to_csv(index=False).encode(),
                    file_name='resultados_previsao.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.write('Erro durante a análise dos dados. Por favor, verifique se o arquivo está no formato correto.')
                st.write(e)

if __name__ == "__main__":
    main()
