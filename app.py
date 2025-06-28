import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os

# --- CONFIGURAÇÕES DA PÁGINA E CARREGAMENTO DO MODELO ---

st.set_page_config(
    page_title="Data App - Saúde Mental",
    page_icon="🧠",
    layout="wide"
)

# URL de download direto do seu modelo no Google Drive
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1E3NGesiaFXVGEOp8t_wBIs8BzijWSVsX'
MODEL_PATH = 'modelo_work_interest.pkl'

# Use @st.cache_resource para carregar o modelo apenas uma vez
@st.cache_resource
def download_and_load_model(model_url, model_path):
    """Baixa o modelo se ele não existir localmente e depois o carrega."""
    if not os.path.exists(model_path):
        st.info("Modelo não encontrado. Baixando do Google Drive... por favor, aguarde.")
        with st.spinner('Baixando modelo...'):
            try:
                r = requests.get(model_url, allow_redirects=True)
                r.raise_for_status()  # Verifica se o download foi bem-sucedido
                with open(model_path, 'wb') as f:
                    f.write(r.content)
                st.success("Download do modelo concluído!")
            except requests.exceptions.RequestException as e:
                st.error(f"Erro ao baixar o modelo: {e}")
                return None

    # Carrega o modelo do arquivo local
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Arquivo do modelo '{model_path}' não encontrado mesmo após a tentativa de download.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")
        return None

# Chama a função para garantir que o modelo seja baixado e carregado
model = download_and_load_model(MODEL_URL, MODEL_PATH)


# --- NAVEGAÇÃO E PÁGINAS ---

opcoes = ['Boas-vindas',
          'Dashboard Interativo',
          'Previsão de Interesse no Trabalho']

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

    # Carrega o dataset a partir do Google Drive
    @st.cache_data
    def load_data():
        """Baixa o dataset do Google Drive se ele não existir localmente."""
        # Link para o dataset no Google Drive (ATUALIZADO)
        DATASET_URL = 'https://drive.google.com/uc?export=download&id=1ASanAI-8GIXbBsek87_WiaHQCM5FMRxA'
        DATASET_PATH = 'mental_health.csv'

        if not os.path.exists(DATASET_PATH):
            st.info("Dataset não encontrado. Baixando do Google Drive... por favor, aguarde.")
            with st.spinner("Baixando dataset..."):
                try:
                    r = requests.get(DATASET_URL, allow_redirects=True)
                    r.raise_for_status()
                    with open(DATASET_PATH, 'wb') as f:
                        f.write(r.content)
                    st.success("Download do dataset concluído!")
                except requests.exceptions.RequestException as e:
                    st.error(f"Erro ao baixar o dataset: {e}")
                    return None
        
        try:
            return pd.read_csv(DATASET_PATH)
        except FileNotFoundError:
            st.error(f"Arquivo '{DATASET_PATH}' não encontrado.")
            return None

    # Carrega os dados
    dados = load_data()

    if dados is not None:
        st.write("Visão geral dos dados brutos:")
        st.dataframe(dados.head())

        st.markdown('---')
        st.subheader("Filtros e Métricas")

        col1, col2, col3 = st.columns(3)

        # Filtros
        genero = col1.selectbox("Gênero", dados['Gender'].unique())
        hist_familiar = col2.selectbox("Histórico Familiar", dados['family_history'].unique())
        ocupacao = col3.selectbox("Ocupação", dados['Occupation'].unique())

        # Aplicando filtros
        filtro = (dados['Gender'] == genero) & (dados['family_history'] == hist_familiar) & (dados['Occupation'] == ocupacao)
        dados_filtrados = dados[filtro]

        if dados_filtrados.empty:
            st.warning("Nenhum dado encontrado para a combinação de filtros selecionada.")
        else:
            col1, col2 = st.columns([1, 2])
            # Métricas
            with col1:
                st.metric('Indivíduos na Seleção', dados_filtrados.shape[0])
                st.metric('Média de Mudanças de Humor (Low=0, Medium=1, High=2)',
                          round(dados_filtrados['Mood_Swings'].replace({'Low': 0, 'Medium': 1, 'High': 2}).mean(), 2))
                st.metric('Procuraram Tratamento', '{:.2%}'.format(dados_filtrados['treatment'].value_counts(normalize=True).get('Yes', 0)))

            # Gráfico
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
    st.markdown("""
        Preencha os campos abaixo com as informações do perfil a ser analisado.
        O modelo irá prever se o interesse no trabalho será **'Sim'** ou **'Não'**.
    """)
    st.markdown('---')

    # Função para pré-processar os dados do usuário EXATAMENTE como no treinamento
    def preprocess_input(user_data):
        # Cria um dicionário para armazenar os dados processados
        processed_data = {}

        # Mapeia cada input do usuário para as colunas dummy
        # A lógica replica o `pd.get_dummies(..., drop_first=True)`

        # Gender (drop_first='Female')
        processed_data['Male'] = 1 if user_data['Gender'] == 'Male' else 0

        # Country (drop_first='Afghanistan') - Adicione mais países se o seu modelo usou
        countries = ['Australia', 'Canada', 'United States']
        for country in countries:
            processed_data[country] = 1 if user_data['Country'] == country else 0

        # Occupation (drop_first='Business')
        occupations = {'occupation:Corporate': 'Corporate', 'occupation:Housewife': 'Housewife', 'occupation:Others': 'Others', 'occupation:Student': 'Student'}
        for col_name, occupation_val in occupations.items():
            processed_data[col_name] = 1 if user_data['Occupation'] == occupation_val else 0

        # self_employed (drop_first='No')
        processed_data['SelfEmployed'] = 1 if user_data['self_employed'] == 'Yes' else 0

        # family_history (drop_first='No')
        processed_data['FamilyHistory'] = 1 if user_data['family_history'] == 'Yes' else 0

        # treatment (drop_first='No')
        processed_data['Treatment'] = 1 if user_data['treatment'] == 'Yes' else 0

        # Days_Indoors (drop_first='0-1 days')
        days_map = {'Days_Indoors:1-14': '1-14 days', 'Days_Indoors:15-30': '15-30 days', 'Days_Indoors:31-60': '31-60 days', 'Days_Indoors:60+': 'More than 2 months', 'Days_Indoors:Go out Every day': 'Go out Every day'}
        for col_name, day_val in days_map.items():
            processed_data[col_name] = 1 if user_data['Days_Indoors'] == day_val else 0

        # Growing_Stress (drop_first='Maybe')
        stress_map = {'Growing_Stress: No': 'No', 'Growing_Stress: Yes': 'Yes'}
        for col_name, stress_val in stress_map.items():
            processed_data[col_name] = 1 if user_data['Growing_Stress'] == stress_val else 0

        # Changes_Habits (drop_first='Maybe')
        habits_map = {'Changes_Habits: No': 'No', 'Changes_Habits: Yes': 'Yes'}
        for col_name, habit_val in habits_map.items():
            processed_data[col_name] = 1 if user_data['Changes_Habits'] == habit_val else 0

        # Mental_Health_History (drop_first='Maybe')
        mhh_map = {'Mental_Health_History: No': 'No', 'Mental_Health_History: Yes': 'Yes'}
        for col_name, mhh_val in mhh_map.items():
            processed_data[col_name] = 1 if user_data['Mental_Health_History'] == mhh_val else 0

        # Mood_Swings (drop_first='High')
        mood_map = {'Mood_Swings: Low': 'Low', 'Mood_Swings: Medium': 'Medium'}
        for col_name, mood_val in mood_map.items():
            processed_data[col_name] = 1 if user_data['Mood_Swings'] == mood_val else 0

        # Coping_Struggles (drop_first='No')
        processed_data['CopingStruggles'] = 1 if user_data['Coping_Struggles'] == 'Yes' else 0

        # Social_Weakness (drop_first='Maybe')
        social_map = {'Social_Weakness: No': 'No', 'Social_Weakness: Yes': 'Yes'}
        for col_name, social_val in social_map.items():
            processed_data[col_name] = 1 if user_data['Social_Weakness'] == social_val else 0

        # mental_health_interview (drop_first='Maybe')
        interview_map = {'mental_health_interview: No': 'No', 'mental_health_interview: Yes': 'Yes'}
        for col_name, interview_val in interview_map.items():
            processed_data[col_name] = 1 if user_data['mental_health_interview'] == interview_val else 0

        # care_options (drop_first='No')
        care_map = {'care_options: Not sure': 'Not sure', 'care_options: Yes': 'Yes'}
        for col_name, care_val in care_map.items():
            processed_data[col_name] = 1 if user_data['care_options'] == care_val else 0

        # Retorna um DataFrame com uma única linha
        return pd.DataFrame(processed_data, index=[0])


    # --- Coleta de Dados do Usuário ---
    st.sidebar.header("Dados para Previsão")

    # Criando os widgets para coletar as informações
    user_inputs = {}
    user_inputs['Gender'] = st.sidebar.selectbox("Gênero", ['Female', 'Male'])
    user_inputs['Country'] = st.sidebar.selectbox("País", ['United States', 'Canada', 'Australia', 'Afghanistan']) # Adicione mais se necessário
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
            # 1. Pré-processar os dados do usuário
            input_df = preprocess_input(user_inputs)

            # 2. Fazer a predição
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            # 3. Exibir o resultado
            st.subheader("Resultado da Predição")
            
            # Assumindo que seu modelo prevê 'Yes' ou 'No'
            resultado = prediction[0]
            probabilidade = prediction_proba[0][list(model.classes_).index(resultado)]

            st.markdown(f"## O interesse no trabalho previsto é: **{resultado.upper()}**")
            st.markdown(f"### Probabilidade da predição: **{probabilidade:.2%}**")

        else:
            st.error("O modelo não está carregado. Não é possível fazer a predição.")

