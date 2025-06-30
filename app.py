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
    st.title('**Data App de Saúde Mental 🧠**')
    st.header('Seja bem-vindo(a)! �')
    st.markdown("""
        Este aplicativo interativo foi desenvolvido para explorar dados sobre saúde mental
        e utilizar um modelo de Machine Learning para realizar previsões.
        **O que você pode fazer aqui?**
        - **Navegar pelo Dashboard:** Explore visualizações e métricas do dataset original.
        - **Realizar Predições:** Use nosso modelo treinado para prever o interesse de um indivíduo
          em seu trabalho com base em um perfil.
        Use o menu na barra lateral à esquerda para navegar entre as seções.
        ---
        *Este é um projeto de estudo e não substitui uma avaliação profissional de saúde.*
    """)

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

# --- PÁGINA 3: PREVISÃO DE INTERESSE NO TRABALHO ---
elif pagina == 'Previsão de Interesse no Trabalho':
    st.header('Modelo Preditivo: Interesse no Trabalho')
    st.markdown("Preencha os campos abaixo com as informações do perfil a ser analisado. O modelo irá prever se o interesse no trabalho será **'Sim'** ou **'Não'**.")
    st.markdown('---')

    def preprocess_input(user_data, expected_cols):
        # Cria um DataFrame com os dados brutos do usuário
        input_df = pd.DataFrame(user_data, index=[0])
        
        # Aplica o One-Hot Encoding
        input_df_processed = pd.get_dummies(input_df, drop_first=True)
        
        # Garante que o dataframe final tenha exatamente as colunas esperadas pelo modelo
        # Adiciona colunas faltantes com valor 0 e remove colunas extras
        final_df = pd.DataFrame(columns=expected_cols)
        final_df = pd.concat([final_df, input_df_processed])
        final_df = final_df.fillna(0)
        
        # Retorna apenas as colunas esperadas na ordem correta
        return final_df[expected_cols]

    # --- Coleta de Dados do Usuário ---
    st.sidebar.header("Dados para Previsão")

    user_inputs = {}
    user_inputs['age'] = st.sidebar.number_input("Idade", min_value=1, max_value=100, value=30)
    user_inputs['Gender'] = st.sidebar.selectbox("Gênero", ['Female', 'Male'])
    user_inputs['Country'] = st.sidebar.selectbox("País", ['United States', 'Canada', 'Australia', 'Belgium', 'Bosnia and Herzegovina', 'Brazil', 'Colombia', 'Costa Rica', 'Croatia', 'Czech Republic', 'Denmark', 'Finland', 'France', 'Georgia', 'Germany', 'Greece', 'India', 'Ireland', 'Israel', 'Italy', 'Mexico', 'Moldova', 'Netherlands', 'New Zealand', 'Nigeria', 'Philippines', 'Poland', 'Portugal', 'Russia', 'Singapore', 'South Africa', 'Sweden', 'Switzerland', 'Thailand', 'United Kingdom'])
    user_inputs['Occupation'] = st.sidebar.selectbox("Ocupação", ['Student', 'Corporate', 'Business', 'Housewife', 'Others'])
    user_inputs['self_employed'] = st.sidebar.radio("É autônomo?", ['No', 'Yes'])
    user_inputs['family_history'] = st.sidebar.radio("Possui histórico familiar de doença mental?", ['No', 'Yes'])
    user_inputs['treatment'] = st.sidebar.radio("Já procurou tratamento?", ['No', 'Yes'])
    user_inputs['Days_Indoors'] = st.sidebar.selectbox("Frequência que fica em ambientes fechados", ['Go out Every day', '1-14 days', '15-30 days', '31-60 days', 'More than 2 months', '0-1 days'])
    user_inputs['Growing_Stress'] = st.sidebar.selectbox("Nível de estresse aumentando?", ['Yes', 'No', 'Maybe'])
    user_inputs['Changes_Habits'] = st.sidebar.selectbox("Houve mudanças de hábitos?", ['Yes', 'No', 'Maybe'])
    user_inputs['Mental_Health_History'] = st.sidebar.selectbox("Possui histórico de saúde mental?", ['No', 'Yes', 'Maybe'])
    user_inputs['Mood_Swings'] = st.sidebar.selectbox("Nível de mudança de humor", ['Low', 'Medium', 'High'])
    user_inputs['Coping_Struggles'] = st.sidebar.radio("Dificuldade em lidar com problemas?", ['No', 'Yes'])
    user_inputs['Social_Weakness'] = st.sidebar.selectbox("Sente-se socialmente fraco?", ['No', 'Yes', 'Maybe'])
    user_inputs['mental_health_interview'] = st.sidebar.selectbox("Faria entrevista sobre saúde mental?", ['No', 'Yes', 'Maybe'])
    user_inputs['care_options'] = st.sidebar.selectbox("Conhece opções de cuidado?", ['No', 'Yes', 'Not sure'])

    if st.button('**APLICAR O MODELO**'):
        if model:
            # A lista exata de colunas que o seu modelo espera, na ordem correta.
            expected_cols = ['Male', 'Belgium', 'Bosnia and Herzegovina', 'Brazil', 'Canada', 'Colombia', 'Costa Rica', 'Croatia', 'Czech Republic', 'Denmark', 'Finland', 'France', 'Georgia', 'Germany', 'Greece', 'India', 'Ireland', 'Israel', 'Italy', 'Mexico', 'Moldova', 'Netherlands', 'New Zealand', 'Nigeria', 'Philippines', 'Poland', 'Portugal', 'Russia', 'Singapore', 'South Africa', 'Sweden', 'Switzerland', 'Thailand', 'United Kingdom', 'United States', 'occupationCorporate', 'occupationHousewife', 'occupationOthers', 'occupationStudent', 'SelfEmployed', 'FamilyHistory', 'Treatment', 'Days_Indoors15-30', 'Days_Indoors31-60', 'Days_IndoorsGo out Every day', 'Days_Indoors60+', 'Growing_Stress No', 'Growing_Stress Yes', 'Changes_Habits No', 'Changes_Habits Yes', 'Mental_Health_History No', 'Mental_Health_History Yes', 'Mood_Swings Low', 'Mood_Swings Medium', 'CopingStruggles', 'Social_Weakness No', 'Social_Weakness Yes', 'mental_health_interview No', 'mental_health_interview Yes', 'care_options Not sure', 'care_options Yes']

            # Pré-processa os dados do usuário, garantindo a correspondência com as colunas esperadas
            input_df_processed = preprocess_input(user_inputs, expected_cols)
            
            # O modelo agora recebe os dados no formato exato que ele espera
            prediction = model.predict(input_df_processed)
            prediction_proba = model.predict_proba(input_df_processed)
            
            st.subheader("Resultado da Predição")
            
            resultado = prediction['prediction_label'][0]
            probabilidade = prediction['prediction_score'][0]
            
            st.markdown(f"## O interesse no trabalho previsto é: **{str(resultado).upper()}**")
            st.markdown(f"### Probabilidade da predição: **{probabilidade:.2%}**")
        else:
            st.error("O modelo não está carregado. Não é possível fazer a predição.")

