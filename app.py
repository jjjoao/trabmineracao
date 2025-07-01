import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os

# --- CONFIGURAÇÕES DA PÁGINA E FUNÇÕES DE DOWNLOAD ---

st.set_page_config(
    page_title="Data App - Saúde Mental",
    page_icon="🧠",
    layout="wide"
)

def download_file_from_google_drive(id, destination):
    """Função robusta para baixar arquivos do Google Drive, lidando com avisos."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# --- CARREGAMENTO DO MODELO ---
MODEL_PATH = 'modelo_final (3).pkl'

@st.cache_resource
def load_model(model_path):
    """Carrega o modelo a partir de um arquivo local."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Arquivo do modelo '{model_path}' não encontrado. Certifique-se de que o nome está correto e que ele foi enviado para o repositório do GitHub.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")
        return None

model = load_model(MODEL_PATH)

# --- CARREGAMENTO DO DATASET PARA O DASHBOARD ---
DATASET_ID = '1ASanAI-8GIXbBsek87_WiaHQCM5FMRxA'
DATASET_PATH = 'mental_health.csv'

@st.cache_data
def load_data(file_id, dataset_path):
    """Baixa e carrega o dataset para o dashboard."""
    if not os.path.exists(dataset_path):
        st.info("Dataset não encontrado. Baixando do Google Drive...")
        with st.spinner("Baixando dataset..."):
            download_file_from_google_drive(file_id, dataset_path)
            st.success("Download do dataset concluído!")
    try:
        return pd.read_csv(dataset_path)
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o dataset: {e}")
        return None

# --- NAVEGAÇÃO E PÁGINAS ---
opcoes = ['Boas-vindas', 'Dashboard Interativo', 'Previsão de Interesse no Trabalho']
pagina = st.sidebar.selectbox('Navegue pelo menu:', opcoes)

# --- PÁGINA 1: BOAS-VINDAS ---
if pagina == 'Boas-vindas':
    st.title('**Data App de Análise de Saúde Mental 🧠**')
    st.header('Descrição do Projeto e Metodologia')

    # Texto atualizado conforme solicitado
    st.markdown("""
        Este conjunto de dados parece conter uma variedade de recursos relacionados à análise de texto, análise de sentimentos e indicadores psicológicos, provavelmente derivados de postagens ou dados de texto. Alguns recursos incluem índices de legibilidade, como o Índice de Legibilidade Automatizado (ARI), o Índice de Coleman Liau e o Nível de Ensino Flesch-Kincaid, bem como pontuações de análise de sentimentos, como sentimentos compostos, negativos, neutros e positivos. Além disso, há recursos relacionados a aspectos psicológicos, como estresse econômico, isolamento, uso de substâncias e estresse doméstico. O conjunto de dados parece abranger uma ampla gama de atributos linguísticos, psicológicos e comportamentais, potencialmente adequados para analisar tópicos relacionados à saúde mental em comunidades online ou dados de texto.

        O conjunto de dados fornece insights valiosos sobre saúde mental, analisando padrões linguísticos, sentimentos e indicadores psicológicos em dados de texto. Pesquisadores e cientistas de dados podem obter uma melhor compreensão de como os problemas de saúde mental se manifestam na comunicação online.

        Com uma ampla gama de recursos, incluindo pontuações de análise de sentimentos e indicadores psicológicos, o conjunto de dados oferece oportunidades para o desenvolvimento de modelos preditivos para identificar ou prever resultados de saúde mental com base em dados textuais.
    """)
    st.markdown("---")
    st.subheader("Construção do Modelo")
    st.markdown("""
        O modelo utilizado é uma combinação de dois modelos: um **Extreme Gradient Boosting** e um **Random Forest**.
        
        Inicialmente, foram removidos os indivíduos que responderam "Maybe" para a variável de interesse `Work_Interest` para facilitar a construção do modelo, já que esses não eram de interesse para o estudo. Foi realizado um pré-processamento dos dados, transformando todas as variáveis categóricas com *n* categorias em *n-1* variáveis, e uma separação dos dados entre treino e teste, sendo 70% para treino e 30% para teste. Além disso, foram feitos 10 folds de validação cruzada com o conjunto de treino.
        
        Assim, foram comparados 15 modelos de machine learning através das métricas Acurácia, AUC, Recall, Precisão, F1-Score, Kappa e MCC. Os modelos de Random Forest e Extreme Gradient Boosting tiveram as melhores performances entre todos os outros em todas as métricas.
        
        Dessa forma, foi utilizado o comando `tune_model`, que realizou 10 folds de validação cruzada para selecionar os melhores hiperparâmetros que maximizem o F1-score dos modelos. Finalmente, foi feita uma combinação (ensemble) dos modelos otimizados, fazendo uma média das probabilidades preditas para determinar quais indivíduos têm interesse no trabalho e quais não têm.
        
        O modelo final foi testado nos 30% dos dados separados para teste, e as métricas para a classificação do modelo foram:
    """)
    
    # Tabela de métricas
    metrics_data = {
        'Métrica': ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC'],
        'Valor': [0.9995, 1.0000, 0.9988, 1.0000, 0.9994, 0.9989, 0.9989]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)


# --- PÁGINA 2: DASHBOARD INTERATIVO ---
elif pagina == 'Dashboard Interativo':
    st.header('Dashboard de Análise do Dataset')
    dados = load_data(DATASET_ID, DATASET_PATH)
    if dados is not None:
        st.write("Visão geral dos dados brutos:")
        st.dataframe(dados.head())
        st.markdown('---')
        st.subheader("Filtros e Métricas")
        col1, col2, col3 = st.columns(3)
        genero = col1.selectbox("Gênero", dados['Gender'].unique())
        hist_familiar = col2.selectbox("Histórico Familiar", dados['family_history'].unique())
        ocupacao = col3.selectbox("Ocupação", dados['Occupation'].unique())
        filtro = (dados['Gender'] == genero) & (dados['family_history'] == hist_familiar) & (dados['Occupation'] == ocupacao)
        dados_filtrados = dados[filtro]
        if dados_filtrados.empty:
            st.warning("Nenhum dado encontrado para a combinação de filtros selecionada.")
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric('Indivíduos na Seleção', dados_filtrados.shape[0])
                mood_mode = dados_filtrados['Mood_Swings'].mode()[0]
                st.metric('Humor Mais Comum', mood_mode)
                st.metric('Procuraram Tratamento', '{:.2%}'.format(dados_filtrados['treatment'].value_counts(normalize=True).get('Yes', 0)))
            with col2:
                fig, ax = plt.subplots()
                sns.countplot(data=dados_filtrados, x='Work_Interest', ax=ax, palette='viridis')
                ax.set_title('Distribuição de Interesse no Trabalho (Filtrado)')
                ax.set_xlabel('Interesse no Trabalho')
                ax.set_ylabel('Contagem')
                st.pyplot(fig)

