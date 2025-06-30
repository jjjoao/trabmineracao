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

    # CORREÇÃO FINAL: Replicando o pré-processamento do notebook EXATAMENTE.
    def preprocess_input(user_data, expected_cols):
        df = pd.DataFrame(user_data, index=[0])

        # Gender
        dummiesg = pd.get_dummies(df['Gender'], drop_first=True)
        df = pd.concat([df, dummiesg], axis=1)

        # Country
        dummiesc = pd.get_dummies(df['Country'], drop_first=True)
        df = pd.concat([df, dummiesc], axis=1)

        # Occupation
        dummieso = pd.get_dummies(df['Occupation'], drop_first=True)
        df = pd.concat([df, dummieso], axis=1)
        df = df.rename(columns={'Corporate': 'occupationCorporate', 'Housewife': 'occupationHousewife', 'Others': 'occupationOthers', 'Student': 'occupationStudent'})

        # self_employed
        dummiesse = pd.get_dummies(df['self_employed'], drop_first=True)
        df = pd.concat([df, dummiesse], axis=1)
        df = df.rename(columns={'Yes': 'SelfEmployed'})

        # family_history
        dummiesfh = pd.get_dummies(df['family_history'], drop_first=True)
        df = pd.concat([df, dummiesfh], axis=1)
        df = df.rename(columns={'Yes': 'FamilyHistory'})

        # treatment
        dummiest = pd.get_dummies(df['treatment'], drop_first=True)
        df = pd.concat([df, dummiest], axis=1)
        df = df.rename(columns={'Yes': 'Treatment'})

        # Days_Indoors
        dummiesdi = pd.get_dummies(df['Days_Indoors'], drop_first=True)
        df = pd.concat([df, dummiesdi], axis=1)
        df = df.rename(columns={'15-30 days': 'Days_Indoors15-30', '31-60 days': 'Days_Indoors31-60', 'Go out Every day': 'Days_IndoorsGo out Every day', 'More than 2 months': 'Days_Indoors60+'})

        # Growing_Stress
        dummiesgs = pd.get_dummies(df['Growing_Stress'], drop_first=True)
        df = pd.concat([df, dummiesgs], axis=1)
        df = df.rename(columns={'No': 'Growing_Stress No', 'Yes': 'Growing_Stress Yes'})

        # Changes_Habits
        dummiesch = pd.get_dummies(df['Changes_Habits'], drop_first=True)
        df = pd.concat([df, dummiesch], axis=1)
        df = df.rename(columns={'No': 'Changes_Habits No', 'Yes': 'Changes_Habits Yes'})

        # Mental_Health_History
        dummiesmhh = pd.get_dummies(df['Mental_Health_History'], drop_first=True)
        df = pd.concat([df, dummiesmhh], axis=1)
        df = df.rename(columns={'No': 'Mental_Health_History No', 'Yes': 'Mental_Health_History Yes'})

        # Mood_Swings
        dummiesms = pd.get_dummies(df['Mood_Swings'], drop_first=True)
        df = pd.concat([df, dummiesms], axis=1)
        df = df.rename(columns={'Low': 'Mood_Swings Low', 'Medium': 'Mood_Swings Medium'})

        # Coping_Struggles
        dummiescs = pd.get_dummies(df['Coping_Struggles'], drop_first=True)
        df = pd.concat([df, dummiescs], axis=1)
        df = df.rename(columns={'Yes': 'CopingStruggles'})

        # Social_Weakness
        dummiessw = pd.get_dummies(df['Social_Weakness'], drop_first=True)
        df = pd.concat([df, dummiessw], axis=1)
        df = df.rename(columns={'No': 'Social_Weakness No', 'Yes': 'Social_Weakness Yes'})

        # mental_health_interview
        dummiesmhi = pd.get_dummies(df['mental_health_interview'], drop_first=True)
        df = pd.concat([df, dummiesmhi], axis=1)
        df = df.rename(columns={'No': 'mental_health_interview No', 'Yes': 'mental_health_interview Yes'})

        # care_options
        dummiesco = pd.get_dummies(df['care_options'], drop_first=True)
        df = pd.concat([df, dummiesco], axis=1)
        df = df.rename(columns={'Not sure': 'care_options Not sure', 'Yes': 'care_options Yes'})
        
        # Alinhamento final para garantir que todas as colunas esperadas existam
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        
        return df[expected_cols]

    # --- Coleta de Dados do Usuário ---
    st.sidebar.header("Dados para Previsão")

    user_inputs = {}
    # Removido 'age' pois não está na lista de features do modelo
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
            
            # Pré-processa os dados do usuário antes de enviar para o modelo
            input_df_processed = preprocess_input(user_inputs, expected_cols)
            
            # O modelo agora recebe os dados já no formato que ele espera
            prediction = model.predict(input_df_processed)
            prediction_proba = model.predict_proba(input_df_processed)
            
            st.subheader("Resultado da Predição")
            
            resultado = prediction['prediction_label'][0]
            probabilidade = prediction['prediction_score'][0]
            
            st.markdown(f"## O interesse no trabalho previsto é: **{str(resultado).upper()}**")
            st.markdown(f"### Probabilidade da predição: **{probabilidade:.2%}**")
        else:
            st.error("O modelo não está carregado. Não é possível fazer a predição.")

