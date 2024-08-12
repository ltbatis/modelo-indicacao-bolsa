import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# Título da aplicação
st.title("Revolucionando a Previsão de Indicação de Bolsas com Machine Learning")

# Introdução com destaque para a análise dos dados
st.write("""
Antes de iniciarmos, queremos destacar o trabalho meticuloso realizado na análise de todas as bases de dados da ONG. 
Utilizando nossa poderosa classe de carregamento de CSVs, fomos capazes de integrar e analisar dados de diversas fontes, 
permitindo-nos mergulhar profundamente em cada detalhe que compõe essa incrível base de conhecimento. 
Essa abordagem cuidadosa nos garante que estamos tomando decisões baseadas nos dados mais completos e precisos possíveis.

Aqui está a classe que desenvolvemos para garantir que todos os dados fossem carregados corretamente:
""")


# Mostrar o código da classe CSVLoader
st.code('''import os
import pandas as pd

class CSVLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.dataframes = {}

        self._load_other_tables()
        self._load_merged_data()

    def _load_other_tables(self):
        outras_tabelas_dir = os.path.join(self.base_dir, 'Outras tabelas')
        for file in os.listdir(outras_tabelas_dir):
            if file.endswith('.csv'):
                table_name = file.replace('.csv', '')
                file_path = os.path.join(outras_tabelas_dir, file)
                self.dataframes[table_name] = pd.read_csv(file_path, low_memory=False)

    def _load_merged_data(self):
        for root, dirs, files in os.walk(self.base_dir):
            if 'Merge' in root:
                for file in files:
                    if file == 'merged_data.csv':
                        table_name = os.path.basename(os.path.dirname(root))
                        file_path = os.path.join(root, file)
                        self.dataframes[table_name] = pd.read_csv(file_path, low_memory=False)

    def __getattr__(self, name):
        if name in self.dataframes:
            return self.dataframes[name]
        raise AttributeError(f"'CSVLoader' object has no attribute '{name}'")
      
# Exemplo de uso:
base_dir = "../bruto/csv_output/Tabelas/"
loader = CSVLoader(base_dir)

df_abatimento = loader.TbAbatimento
df_centro_resultado = loader.TbCentroResultado
''', language='python')

st.write("""
Através dessa classe, pudemos garantir que cada tabela, cada detalhe, fosse levado em conta na construção do nosso modelo. 
Agora, vamos explorar os dados e mostrar como conseguimos extrair o melhor dessas informações para prever quais alunos são 
os melhores candidatos a receber uma bolsa de estudos.
""")

# Introdução
st.write("""
Imagine poder identificar, com precisão, quais alunos estão prontos para receber uma bolsa de estudos. 
Nosso modelo de machine learning não é apenas uma ferramenta, é a chave para transformar o futuro de estudantes talentosos.
Hoje, vamos te guiar pelo processo que usamos para chegar ao modelo ideal, mostrando cada passo da nossa jornada - 
desde os primeiros esboços até a criação do que consideramos o modelo definitivo.
""")

# Mostrar o código para leitura dos dados
st.subheader("Passo 1: Conhecendo os Dados")
st.code('''import pandas as pd

filename = "data/PEDE_PASSOS_DATASET_FIAP.csv"
df = pd.read_csv(filename, sep=';')

df.head()''', language='python')

# Executar o código
filename = "data/PEDE_PASSOS_DATASET_FIAP.csv"
df = pd.read_csv(filename, sep=';')
st.write("Vamos começar conhecendo o coração do nosso modelo - os dados que usamos para treinar nossa solução. Aqui estão as primeiras linhas do dataset:")
st.dataframe(df.head())

# Mostrar o código para preparação dos dados
st.subheader("Passo 2: Preparação dos Dados - O Alicerce do Sucesso")
st.code('''df = df.dropna(subset=['INDICADO_BOLSA_2022'])
df['INDICADO_BOLSA_2022'] = df['INDICADO_BOLSA_2022'].map({'Sim': 1, 'Não': 0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop('INDICADO_BOLSA_2022', axis=1)
y = df['INDICADO_BOLSA_2022']

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)''', language='python')

# Executar o código de preparação
df = df.dropna(subset=['INDICADO_BOLSA_2022'])
df['INDICADO_BOLSA_2022'] = df['INDICADO_BOLSA_2022'].map({'Sim': 1, 'Não': 0})
df = pd.get_dummies(df, drop_first=True)
X = df.drop('INDICADO_BOLSA_2022', axis=1)
y = df['INDICADO_BOLSA_2022']
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
st.write("Dados bem preparados são a base de qualquer modelo de sucesso. Aqui, cuidamos de cada detalhe para garantir que nada escape do radar do nosso algoritmo.")

# Mostrar o código para divisão e treinamento do modelo
st.subheader("Passo 3: Primeiros Passos - Testando com Regressão Logística")
st.code('''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)''', language='python')

# Executar o código de treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Mostrar o código para avaliação do modelo
st.subheader("Passo 4: Avaliando o Primeiro Modelo - Regressão Logística")
st.code('''y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Acurácia: {accuracy:.2f}")
st.write("Relatório de Classificação:")
st.json(report)''', language='python')

# Executar o código de avaliação
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)

# Exibir os resultados
st.write(f"Resultado: Acurácia de {accuracy:.2f}.")
st.write("Relatório de Classificação:")
st.json(report)

# Explicação sobre o modelo
st.write("""
Nosso primeiro passo foi promissor! A Regressão Logística nos deu uma boa base, 
mas sabemos que podemos ir além. 
O recall para a classe minoritária mostrou que ainda há espaço para melhorias.
""")

# Adicionando o Modelo Random Forest

# Mostrar o código para o modelo Random Forest
st.subheader("Passo 5: Elevando o Nível - Implementando o Random Forest")
st.code('''# Divisão e normalização dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Balanceamento das classes usando SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Treinamento do modelo Random Forest
model_rf = RandomForestClassifier(class_weight={0: 1, 1: 3}, random_state=42)
model_rf.fit(X_train_res, y_train_res)''', language='python')

# Executar o código de Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model_rf = RandomForestClassifier(class_weight={0: 1, 1: 3}, random_state=42)
model_rf.fit(X_train_res, y_train_res)

# Mostrar o código para avaliação do modelo Random Forest
st.subheader("Passo 6: Avaliando o Random Forest - Um Salto de Qualidade")
st.code('''# Avaliação do modelo
y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

# Avaliação com ROC-AUC
y_prob_rf = model_rf.predict_proba(X_test)[:, 1]
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

# Exibir os resultados
st.write(f"Acurácia: {accuracy_rf:.2f}")
st.write(f"ROC-AUC: {roc_auc_rf:.2f}")
st.write("Relatório de Classificação:")
st.json(report_rf)''', language='python')

# Executar o código de avaliação do Random Forest
y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
y_prob_rf = model_rf.predict_proba(X_test)[:, 1]
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

# Exibir os resultados
st.write(f"Resultado: Acurácia de {accuracy_rf:.2f}, com uma impressionante ROC-AUC de {roc_auc_rf:.2f}.")
st.write("Relatório de Classificação:")
st.json(report_rf)

# Explicação sobre o modelo Random Forest
st.write("""
O Random Forest nos mostrou que estamos no caminho certo! 
Com uma ROC-AUC elevada, estamos capturando mais nuances nos dados, 
identificando melhor aqueles alunos que realmente merecem ser indicados.
Mas não vamos parar por aqui - nosso objetivo é alcançar a perfeição!
""")

import numpy as np

# Passo 7: Exemplo de Predição com o Modelo Treinado
st.subheader("Passo 7: Teste de Predição com Novos Dados")

st.write("Agora que temos um modelo poderoso, vamos testar sua capacidade de prever se um aluno deve ser indicado para a bolsa de estudos.")

# Função para gerar valores aleatórios baseados em exemplos que foram classificados como "Indicado"
def gerar_valores_para_indicado():
    indicados = df[df['INDICADO_BOLSA_2022'] == 1]  # Filtra apenas os indicados
    return {feature: np.random.uniform(indicados[feature].min(), indicados[feature].max()) for feature in indicados.drop(columns=['INDICADO_BOLSA_2022']).columns}

# Estado inicial dos valores baseados em indicados
if 'valores_aleatorios' not in st.session_state:
    st.session_state['valores_aleatorios'] = gerar_valores_para_indicado()

# Botão para gerar novos valores aleatórios baseados em indicados
if st.button("Gerar Valores para Indicado"):
    st.session_state['valores_aleatorios'] = gerar_valores_para_indicado()

# Permitir que o usuário insira valores para um novo exemplo com valores baseados em indicados já preenchidos
inputs = {}
for feature, valor in st.session_state['valores_aleatorios'].items():
    inputs[feature] = st.number_input(f"{feature}", value=valor)

# Converter os inputs para um DataFrame com as mesmas colunas do modelo treinado
new_data = pd.DataFrame([inputs])

# Fazer a predição
if st.button("Prever"):
    # Aplicar o mesmo scaler utilizado no treinamento
    new_data_scaled = scaler.transform(new_data)
    prediction = model_rf.predict(new_data_scaled)
    prediction_prob = model_rf.predict_proba(new_data_scaled)

    st.write(f"Predição: {'Indicado' if prediction[0] == 1 else 'Não Indicado'}")
    st.write(f"Confiança da Predição: {prediction_prob[0][prediction[0]] * 100:.2f}%")