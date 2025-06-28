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
    
    # Salva o conteúdo baixado no arquivo de destino
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# --- CARREGAMENTO DO MODELO ---
MODEL_ID = '1E3NGesiaFXVGEOp8t_wBIs8BzijWSVsX'
MODEL_PATH = 'modelo_work_interest.pkl'

@st.cache_resource
def load_model(file_id, model_path):
    """Baixa e carrega o modelo de ML."""
    if not os.path.exists(model_path):
        st.info("Modelo não encontrado. Baixando do Google Drive...")
        with st.spinner('Baixando modelo...'):
            download_file_from_google_drive(file_id, model_path)
            
            # Verificação de segurança para garantir que o arquivo baixado não é uma página de erro
            if os.path.exists(model_path):
                file_size_kb = os.path.getsize(model_path) / 1024
                if file_size_kb < 100: # Se o arquivo for muito pequeno, provavelmente é um erro.
                    os.remove(model_path) # Remove o arquivo inválido
                    st.error("Falha no download. O arquivo baixado é muito pequeno, o que sugere um erro de permissão ou um link inválido no Google Drive. Por favor, verifique se o link de compartilhamento está como 'Qualquer pessoa com o link'.")
                    return None
            st.success("Download do modelo concluído!")
            
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        # Este erro não deve acontecer se a lógica acima estiver correta, mas é uma boa prática mantê-lo.
        st.error(f"Arquivo do modelo '{model_path}' não foi encontrado.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")
        return None

model = load_model(MODEL_ID, MODEL_PATH)

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
    st.header('Seja bem-vindo(a)! 😀')
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
                # Métrica corrigida para mostrar a moda (valor mais comum) em vez da média
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

    def preprocess_input(user_data):
        processed_data = {}
        processed_data['Male'] = 1 if user_data['Gender'] == 'Male' else 0
        countries = ['Australia', 'Canada', 'United States']
        for country in countries:
            processed_data[country] = 1 if user_data['Country'] == country else 0
        occupations = {'occupation:Corporate': 'Corporate', 'occupation:Housewife': 'Housewife', 'occupation:Others': 'Others', 'occupation:Student': 'Student'}
        for col_name, occupation_val in occupations.items():
            processed_data[col_name] = 1 if user_data['Occupation'] == occupation_val else 0
        processed_data['SelfEmployed'] = 1 if user_data['self_employed'] == 'Yes' else 0
        processed_data['FamilyHistory'] = 1 if user_data['family_history'] == 'Yes' else 0
        processed_data['Treatment'] = 1 if user_data['treatment'] == 'Yes' else 0
        days_map = {'Days_Indoors:1-14': '1-14 days', 'Days_Indoors:15-30': '15-30 days', 'Days_Indoors:31-60': '31-60 days', 'Days_Indoors:60+': 'More than 2 months', 'Days_Indoors:Go out Every day': 'Go out Every day'}
        for col_name, day_val in days_map.items():
            processed_data[col_name] = 1 if user_data['Days_Indoors'] == day_val else 0
        stress_map = {'Growing_Stress: No': 'No', 'Growing_Stress: Yes': 'Yes'}
        for col_name, stress_val in stress_map.items():
            processed_data[col_name] = 1 if user_data['Growing_Stress'] == stress_val else 0
        habits_map = {'Changes_Habits: No': 'No', 'Changes_Habits: Yes': 'Yes'}
        for col_name, habit_val in habits_map.items():
            processed_data[col_name] = 1 if user_data['Changes_Habits'] == habit_val else 0
        mhh_map = {'Mental_Health_History: No': 'No', 'Mental_Health_History: Yes': 'Yes'}
        for col_name, mhh_val in mhh_map.items():
            processed_data[col_name] = 1 if user_data['Mental_Health_History'] == mhh_val else 0
        mood_map = {'Mood_Swings: Low': 'Low', 'Mood_Swings: Medium': 'Medium'}
        for col_name, mood_val in mood_map.items():
            processed_data[col_name] = 1 if user_data['Mood_Swings'] == mood_val else 0
        processed_data['CopingStruggles'] = 1 if user_data['Coping_Struggles'] == 'Yes' else 0
        social_map = {'Social_Weakness: No': 'No', 'Social_Weakness: Yes': 'Yes'}
        for col_name, social_val in social_map.items():
            processed_data[col_name] = 1 if user_data['Social_Weakness'] == social_val else 0
        interview_map = {'mental_health_interview: No': 'No', 'mental_health_interview: Yes': 'Yes'}
        for col_name, interview_val in interview_map.items():
            processed_data[col_name] = 1 if user_data['mental_health_interview'] == interview_val else 0
        care_map = {'care_options: Not sure': 'Not sure', 'care_options: Yes': 'Yes'}
        for col_name, care_val in care_map.items():
            processed_data[col_name] = 1 if user_data['care_options'] == care_val else 0
        return pd.DataFrame(processed_data, index=[0])

    st.sidebar.header("Dados para Previsão")
    user_inputs = {}
    user_inputs['Gender'] = st.sidebar.selectbox("Gênero", ['Female', 'Male'])
    user_inputs['Country'] = st.sidebar.selectbox("País", ['United States', 'Canada', 'Australia', 'Afghanistan'])
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
            input_df = preprocess_input(user_inputs)
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            st.subheader("Resultado da Predição")
            resultado = prediction[0]
            probabilidade = prediction_proba[0][list(model.classes_).index(resultado)]
            st.markdown(f"## O interesse no trabalho previsto é: **{resultado.upper()}**")
            st.markdown(f"### Probabilidade da predição: **{probabilidade:.2%}**")
        else:
            st.error("O modelo não está carregado. Não é possível fazer a predição.")
