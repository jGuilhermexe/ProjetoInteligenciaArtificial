import os 
import pandas as pd  # Importação da biblioteca pandas para que seja possível a leitura e manipulação de tabelas (DataFrame)
import numpy as np  # Importação da biblioteca numpy para operações numéricas básicas

# Ferramentas de validação e busca de parâmetros do scikit-learn
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importando os 5 classificadores: Árvore de Decisão, KNN, Naive Bayes, Regressão Logística e MLP
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 1) PRÉ-PROCESSAMENTO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("\n  Iniciando o pré-processamento do detect_dataset.csv \n")

#Carregar dataset
df_detect = pd.read_csv("detect_dataset.csv")

# Remove colunas vazias
# O parâmetro 'errors='ignore'' evita erro caso as colunas "Unnamed: 7" e "Unnamed: 8" já tenham sido removidas ou não existam.
df_detect = df_detect.drop(columns=["Unnamed: 7", "Unnamed: 8"], errors='ignore')

# Início da Impressão de Informações
print("Shape do DataFrame após carregar e remover colunas vazias:", df_detect.shape)
print("\nPrimeiras 5 registros (com colunas vazias removidas):")
print(df_detect.head())

# Informações básicas sobre o dataset
print("\nTipos das colunas:")
print(df_detect.dtypes)
print("\nValores ausentes por coluna (antes da limpeza):")
print(df_detect.isnull().sum())
print("\nTotal de linhas duplicadas (antes da limpeza):", df_detect.duplicated().sum())

# Limpeza para remover duplicadas e nulos
df_clean = df_detect.drop_duplicates()
df_clean = df_clean.dropna()
print("\nShape após remover duplicadas:", df_clean.shape)
print("Shape após remover nulos:", df_clean.shape)

# Distribuição da classe (detectada automaticamente)
coluna_classe = df_detect.columns[0] 
print(f"\nDistribuição da classe ({coluna_classe}):")
print(df_detect[coluna_classe].value_counts())

# Salvar bases
df_detect.to_csv("detect_dataset_original.csv", index=False)
df_clean.to_csv("detect_dataset_processada.csv", index=False)
print(f"\n Base original salva como detect_dataset_original.csv")
print(f"\n Base processada salva como detect_dataset_processada.csv")
print("\n PRÉ-PROCESSAMENTO CONCLUÍDO \n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2) CARREGAMENTO DO DATASET ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Iniciando carregamento do dataset processado...")

# Carregando o dataset já processado.
df = pd.read_csv("detect_dataset_processada.csv")

# Fazendo a separação dos atributos (X) e rótulos (y):
# - X deve contem apenas as features (dados de entrada numéricos)
# - y é a coluna que contém as classes que queremos prever
X = df.drop(columns=["Output (S)"])  # todas as colunas exceto a coluna de saída
y = df["Output (S)"]  # coluna principal

# Notificações importantes no terminal pra verificar que o dataset carregou corretamente:
print("Dataset processado carregado com sucesso!")
print("Primeiras linhas do DataFrame:")
print(df.head(), "\n")
print("Formato do DataFrame (linhas, colunas):", df.shape)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 3) DEFINIÇÃO DOS MODELOS E PARÂMETROS (mínimo 3 combinações) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Nesta versão, **não teremos parâmetro nenhum**, pois isso será feito pelo seu amigo.
# Mantemos APENAS os classificadores, como parte de "Uso dos Algoritmos".

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 3) DEFINIÇÃO DOS MODELOS E PARÂMETROS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 3.1) Definição dos classificadores base (adicionado random_state para reprodutibilidade)
modelos = {
    "Decision Tree": DecisionTreeClassifier(random_state=42), 
    "KNN": KNeighborsClassifier(), 
    "Naive Bayes": GaussianNB(),
    "Regressão Logística": LogisticRegression(max_iter=500, random_state=42), 
    "MLP": MLPClassifier(max_iter=1000, random_state=42)
}

# 3.2) Definição dos grids de parâmetros para OTIMIZAÇÃO (Critério 1.2)

# Grid para KNN (mínimo de 3 combinações garantido)
params_knn = {
    'clf__n_neighbors': [3, 5, 7, 9],  # K de 3, 5, 7 e 9
    'clf__weights': ['uniform', 'distance'] # Dois tipos de pesos
}

# Grid para Decision Tree (mínimo de 3 combinações garantido)
params_dt = {
    'clf__criterion': ['gini', 'entropy'], # Critérios
    'clf__max_depth': [5, 10, None],      # Profundidade (5, 10 ou ilimitada)
    'clf__min_samples_split': [2, 10]     # Mínimo de amostras para um split
}

# Grid para MLP (mínimo de 3 combinações garantido)
params_mlp = {
    'clf__hidden_layer_sizes': [(50,), (100, 50)], # Estrutura das camadas ocultas
    'clf__activation': ['relu', 'tanh'],           # Funções de ativação
    'clf__alpha': [0.0001, 0.01]                   # Regularização L2
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 4) FUNÇÃO DE AVALIAÇÃO EM 10-FOLD CV ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def avaliar_modelo(nome, modelo, X, y):
    # Avalia um modelo usando Stratified 10-fold CV.
    #Calcula métricas fold-a-fold pra obter média e desvio padrão com sucesso
    # Parâmetros:
    # - nome: string com o nome do modelo (apenas para impressão)
    # - modelo: objeto classificador do sklearn (Como por exexmplo: DecisionTreeClassifier())
    # - X, y: dados e rótulos (Biblioteca pandas e DataFrame / Series)

    print(f"\n Avaliando: {nome}")

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 5) OTIMIZAÇÃO DE PARÂMETROS (GridSearch) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def otimizar_modelo(nome, modelo, parametros, X, y):
    print(f"\n✨ Otimizando Hiperparâmetros para: {nome} (GridSearch)")
    
    # Cria o Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", modelo) 
    ])
    
    # Configura o GridSearchCV (usa 10-Fold CV interno)
    grid_search = GridSearchCV(
        estimator=pipeline, 
        param_grid=parametros, 
        scoring='f1_weighted', # Usamos F1-Score Ponderado como a métrica principal para o GridSearch
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), 
        verbose=1, 
        n_jobs=-1 # Usa todos os núcleos do processador para acelerar
    )
    
    # Executa o Grid Search
    grid_search.fit(X, y)
    
    # Imprime os resultados
    print(f"   🏆 Melhor F1-Score (média CV): {grid_search.best_score_:.4f}")
    print(f"   ⚙️  Melhores Parâmetros: {grid_search.best_params_}")
    
    # Retorna o melhor modelo encontrado
    return grid_search.best_estimator_['clf']


# --- Execução da Otimização ---
melhores_modelos_otimizados = {}
print("\n" + "="*50)
print("INICIANDO A OTIMIZAÇÃO DE PARÂMETROS (Critério 1.2)")
print("="*50)

# 1. Otimização para KNN
melhores_modelos_otimizados["KNN (Otimizado)"] = otimizar_modelo(
    "KNN", KNeighborsClassifier(), params_knn, X, y
)

# 2. Otimização para Decision Tree
melhores_modelos_otimizados["Decision Tree (Otimizada)"] = otimizar_modelo(
    "Decision Tree", DecisionTreeClassifier(random_state=42), params_dt, X, y
)

# 3. Otimização para MLP
melhores_modelos_otimizados["MLP (Otimizado)"] = otimizar_modelo(
    "MLP", MLPClassifier(max_iter=1000, random_state=42), params_mlp, X, y
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 6) EXECUTAR TODOS OS MODELOS (Originais e Otimizados) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Incluir os modelos otimizados na lista de modelos a serem avaliados
modelos.update(melhores_modelos_otimizados) 

print("\n" + "="*50)
print("AVALIAÇÃO FINAL (Originais e Otimizados)")
print("="*50)

# Avaliar todos os modelos (originais + otimizados)
for nome, modelo in modelos.items():
    avaliar_modelo(nome, modelo, X, y)