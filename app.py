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
        st.write('Olá, seja bem vindo ao DataCurrencie Bank')

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
        sns.histplot(df['tempo_emprego'], bins=30, kde=True, color="#9c88ff")  # Usando roxo para o histograma
        plt.title('')
        plt.xlabel('Anos de Emprego')
        plt.ylabel('Número de Clientes')
        st.pyplot(plt)

    def plot_income_distribution(df):
        # Aplicando a transformação logarítmica à renda
        df['renda_log'] = np.log(df['renda'] + 1)
        
        plt.figure(figsize=(8, 5))
        sns.histplot(df['renda_log'], bins=30, kde=True, color="#9c88ff")  # Amarelo claro para o histograma
        plt.title('')
        plt.xlabel('Log da Renda')
        plt.ylabel('Número de Clientes')
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
    st.subheader('Relações entre variáveis')

    def plot_analises_comparativas(data):

        # Verificar se todas as colunas necessárias estão presentes no DataFrame
        colunas_necessarias = ['posse_de_veiculo', 'posse_de_imovel', 'idade', 'qtd_filhos', 'educacao', 'renda', 'tipo_renda', 'tempo_emprego']
        for col in colunas_necessarias:
            if col not in data.columns:
                raise ValueError(f"Falta a coluna '{col}' no DataFrame fornecido.")

        # Configurar subplots
        fig, axs = plt.subplots(3, 2, figsize=(20, 18))
        fig.tight_layout(pad=5.0)

        # Gráfico 1: Relação entre Posse de Veículo e Posse de Imóvel
        cross_tab = pd.crosstab(data['posse_de_veiculo'], data['posse_de_imovel'])
        cross_tab.plot(kind='bar', stacked=True, ax=axs[0, 0])
        axs[0, 0].set_title('Relação entre Posse de Veículo e Posse de Imóvel')
        axs[0, 0].set_xlabel('Posse de Veículo')
        axs[0, 0].set_ylabel('Contagem')
        axs[0, 0].legend(title='Posse de Imóvel', labels=['Sim', 'Não'])

        # Gráfico 2: Média de Filhos por Faixa Etária
        limites = [20, 30, 40, 50, 60, 70]
        faixas_etarias = ["20-29", "30-39", "40-49", "50-59", "60+"]
        data['faixa_etaria'] = pd.cut(data['idade'], bins=limites, labels=faixas_etarias, right=False)
        media_filhos_por_faixa = data.groupby('faixa_etaria')['qtd_filhos'].mean()
        media_filhos_por_faixa.plot(kind='bar', color='purple', alpha=0.7, ax=axs[0, 1])
        axs[0, 1].set_title('Média de Filhos por Faixa Etária')
        axs[0, 1].set_xlabel('Faixa Etária')
        axs[0, 1].set_ylabel('Média de Filhos')

        # Gráfico 3: Média da Renda Anual por Nível de Educação
        media_renda_por_educacao = data.groupby('educacao')['renda'].mean().reset_index()
        sns.barplot(x='renda', y='educacao', data=media_renda_por_educacao, palette='Set3', ax=axs[1, 0])
        axs[1, 0].set_title('Média da Renda Anual por Nível de Educação')

        # Gráfico 4: Relação entre Tipo de Renda e Tempo de Emprego
        sns.boxplot(x='tipo_renda', y='tempo_emprego', data=data, palette='Set3', ax=axs[1, 1])
        axs[1, 1].set_title('Relação entre Tipo de Renda e Tempo de Emprego')
        axs[1, 1].set_xlabel('Tipo de Renda')
        axs[1, 1].set_ylabel('Tempo de Emprego (Anos)')
        axs[1, 1].tick_params(axis='x', rotation=45)

        # Gráfico 5: Distribuição de Posse de Imóvel por Faixa de Renda Anual
        faixas_renda = [0, 4000, 6000, 8000, float('inf')]
        categorias_renda = ['0-4000', '4000-6000', '6000-8000', '8000+']
        data['faixa_renda'] = pd.cut(data['renda'], bins=faixas_renda, labels=categorias_renda, right=False)
        tabela_cruzada = pd.crosstab(data['faixa_renda'], data['posse_de_imovel'])
        tabela_cruzada.plot(kind='bar', stacked=True, edgecolor='white', width=0.7, ax=axs[2, 0])
        axs[2, 0].set_title('Distribuição de Posse de Imóvel por Faixa de Renda Anual')
        axs[2, 0].set_xlabel('Faixa de Renda Anual')
        axs[2, 0].set_ylabel('Contagem')

        # Removendo o gráfico vazio
        axs[2, 1].axis('off')

        st.pyplot(fig)

    plot_analises_comparativas(df)
    # 2ª Seção: Upload e Análise da Base de Testes
    st.header('2. Avaliação dos Dados')
    
    # Carregando os modelos (substitua com o caminho correto para seus modelos)
    logistic_model = load_model('caminho/para/seu/modelo_logistico.pkl')
    lgbm_model = load_model('caminho/para/seu/modelo_lightgbm.pkl')

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
