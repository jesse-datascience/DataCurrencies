# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

# Função para carregar modelos
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Função principal do Streamlit
def main():
    st.title('Análise de Crédito - DataCurrencies')

    # Divide a interface do usuário em duas colunas
    col1, col2 = st.columns(2)

    # Primeira sessão: Detalhes da Base de Dados
    with col1:
        st.header('1. Detalhes da Base de Dados Original')
        st.write("""
        Características dos clientes utilizados para treinar os modelos de análise de crédito.
        """)
        
        # Você pode adicionar detalhes sobre a quantidade de dados, distribuições, estatísticas descritivas, etc.
        # Abaixo, estou apenas listando as variáveis para simplicidade.
        st.subheader('Variáveis Qualitativas:')
        qualitative_vars = ['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia', 'mau']
        st.write(qualitative_vars)

        st.subheader('Variáveis Quantitativas:')
        quantitative_vars = ['qtd_filhos', 'idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda']
        st.write(quantitative_vars)

    # Segunda sessão: Upload e Análise da Base de Testes
    with col2:
        st.header('2. Avaliação com a Base de Testes')
        st.write("""
        Faça o upload de sua base de testes no formato CSV e selecione o modelo para realizar a análise.
        """)

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
