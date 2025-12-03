import os 
import pandas as pd  # Importação da biblioteca pandas para que seja possível a leitura e manipulação de tabelas (DataFrame)
import numpy as np  # Importação da biblioteca numpy para operações numéricas básicas

# Ferramentas de validação e busca de parâmetros do scikit-learn
from sklearn.model_selection import StratifiedKFold 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importando os 5 classificadores: Árvore de Decisão, KNN, Naive Bayes, Regressão Logística e MLP
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 1) CARREGAMENTO DO DATASET ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Lê o arquivo CSV fornecido pelo site do dataset para um DataFrame do pandas.
df = pd.read_csv("detect_dataset.csv")


# Função para remover colunas vazias que vem diretamente do arquivo csv
# Se essas colunas não existirem, o drop causará um erro — use df.columns para checar.
df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"])

# Fazendo a separação dos atributos (X) e rótulos (y):
# - X deve contem apenas as features (dados de entrada numéricos)
# - y é a coluna que contém as classes que queremos prever

X = df.drop(columns=["Output (S)"])  # todas as colunas exceto a coluna de saída
y = df["Output (S)"]  # coluna principal

# Notificações importantes no terminal pra verificar que o dataset carregou corretamente:
print("Dataset carregado com sucesso!")
print("Primeiras linhas do DataFrame:")
print(df.head(), "\n")
print("Formato do DataFrame (linhas, colunas):", df.shape)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2) DEFINIÇÃO DOS MODELOS E PARÂMETROS (mínimo 3 combinações) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Nesta versão, **não teremos parâmetro nenhum**, pois isso será feito pelo seu amigo.
# Mantemos APENAS os classificadores, como parte de "Uso dos Algoritmos".

modelos = {
    "Decision Tree": DecisionTreeClassifier(),  # Árvore de Decisão
    "KNN": KNeighborsClassifier(),              # KNN
    "Naive Bayes": GaussianNB(),                # Naive Bayes
    "Regressão Logística": LogisticRegression(max_iter=500),  # Regressão Logística
    "MLP": MLPClassifier(max_iter=1000)         # MLP
}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 3) FUNÇÃO DE AVALIAÇÃO EM 10-FOLD CV ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def avaliar_modelo(nome, modelo, X, y):
    # Avalia um modelo usando Stratified 10-fold CV.
    #Calcula métricas fold-a-fold pra obter média e desvio padrão com sucesso

    # Parâmetros:
    # - nome: string com o nome do modelo (apenas para impressão)
    # - modelo: objeto classificador do sklearn (Como por exexmplo: DecisionTreeClassifier())
    # - X, y: dados e rótulos (Biblioteca pandas e DataFrame / Series)


    print(f"\n🔵 Avaliando: {nome}")

    # Pipeline que aplica StandardScaler (normalização) e depois o classificador.
    # A ordem é importante: primeiro transformações (scaler), depois o estimador (clf).
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # step 'scaler'
        ("clf", modelo)                # step 'clf' que contém o classificador
    ])

    # Criação do validador estratificado: garante que tenha a mesma proporção de classes em cada fold.
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Listas para armazenar métricas obtidas fold a fold.
    accs = []
    precs = []
    recalls = []
    f1s = []

    # Aqui iteramos explicitamente sobre os folds para calcular as métricas manualmente.
    for train_idx, test_idx in cv.split(X, y):

        # Divisão manual entre treino e teste usando os índices do fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Treinando o pipeline completo neste fold
        pipeline.fit(X_train, y_train)

        # Prevendo apenas nas amostras do fold de teste
        preds = pipeline.predict(X_test)

        # Calculando métricas para este fold e armazenar
        accs.append(accuracy_score(y_test, preds))
        precs.append(precision_score(y_test, preds))
        recalls.append(recall_score(y_test, preds))
        f1s.append(f1_score(y_test, preds))

    # Impressão dos resultados consolidados: médias e desvios
    print(f"Acurácia: média={np.mean(accs):.4f} | desvio={np.std(accs):.4f}")
    print(f"Precisão: média={np.mean(precs):.4f} | desvio={np.std(precs):.4f}")
    print(f"Recall:   média={np.mean(recalls):.4f} | desvio={np.std(recalls):.4f}")
    print(f"F1-Score: média={np.mean(f1s):.4f} | desvio={np.std(f1s):.4f}")


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  4) EXECUTAR OS 5 MODELOS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Função para executar os 5 modelos:
for nome, modelo in modelos.items():
    avaliar_modelo(nome, modelo, X, y)

#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  PRÉ-PROCESSAMENTO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import pandas as pd

print("\n INICIANDO PRÉ-PROCESSAMENTO \n")

#Carregar dataset
df = pd.read_csv("classData.csv")

print("Shape original:", df.shape)
print("\nPrimeiros 5 registros:")
print(df.head())

#Informações básicas sobre o dataset
print("\nTipos das colunas:")
print(df.dtypes)

print("\nValores ausentes por coluna:")
print(df.isnull().sum())

print("\nTotal de linhas duplicadas:", df.duplicated().sum())

#Limpeza para remover duplicadas e nulos
df_clean = df.drop_duplicates()
df_clean = df_clean.dropna()

print("\nShape após remover duplicadas:", df_clean.shape)
print("Shape após remover nulos:", df_clean.shape)

# Distribuição da classe (detectada automaticamente)
coluna_classe = df.columns[-1]
print(f"\nDistribuição da classe ({coluna_classe}):")
print(df[coluna_classe].value_counts())

#Salvar bases
df.to_csv("classData_original.csv", index=False)
df_clean.to_csv("classData_processada.csv", index=False)

print("\n Base original limpa salva como classData_original.csv")
print("\n Base processada salva como classData_processada.csv")
print("\n PRÉ-PROCESSAMENTO CONCLUÍDO \n")



