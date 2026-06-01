import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import OneHotEncoder


def make_one_hot_encoder():
    """Compatibilidade entre versões novas e antigas do scikit-learn."""
    try:
        return OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(drop=None, sparse=False, handle_unknown="ignore")


def evaluate_kmeans(X, title, k_min=2, k_max=9):
    metrics = []

    for k in range(k_min, k_max + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        metrics.append({
            "k": k,
            "silhouette_score": silhouette_score(X, labels),
            "davies_bouldin_index": davies_bouldin_score(X, labels)
        })

    metrics_df = pd.DataFrame(metrics)

    print(f"\nMétricas por número de clusters - {title}:")
    print(metrics_df)

    return metrics_df


def build_demographic_features(data):
    # Age entra como variável numérica, sem padronização.
    age_feature = data[["Age"]].astype(float).reset_index(drop=True)

    # Gender entra como variáveis categóricas via One-Hot Encoding..
    categorical_cols = ["Gender"]
    encoder = make_one_hot_encoder()
    encoded = encoder.fit_transform(data[categorical_cols].astype(str))
    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    categorical_features = pd.DataFrame(
        encoded,
        columns=encoded_cols,
        index=data.index
    ).reset_index(drop=True)

    return pd.concat([age_feature, categorical_features], axis=1)


# -----------------------------------------------------------------------------
# Leitura e limpeza dos dados
# -----------------------------------------------------------------------------


df = pd.read_excel("RTVue_20221110_MLClass.xlsx")

features = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']
demographic_features = ["Age", "Gender"]

original_rows = len(df)

# 1. Remove idade inválida
data = df[df['Age'].between(0, 120)].copy()
removed_invalid_age = original_rows - len(data)

# 2. Transforma outliers em NaN
for col in features:
    data.loc[~data[col].between(28, 80), col] = pd.NA

# 3. Conta quantos valores ausentes/outliers cada linha possui
missing_count = data[features].isna().sum(axis=1)

# 4. Remove linhas com mais de um valor ausente/outlier
before_missing_clean = len(data)
data = data[missing_count <= 1].copy()
removed_missing_or_outlier = before_missing_clean - len(data)

# 5. Preenche o que restou ausente com a mediana da coluna
for col in features:
    data[col] = data[col].fillna(data[col].median())

final_rows = len(data)

""" print("Linhas originais:", original_rows)
print("Removidas por idade inválida:", removed_invalid_age)
print("Removidas por mais de um ausentes/outliers:", removed_missing_or_outlier)
print("Linhas finais:", final_rows)
print("Total removido:", original_rows - final_rows) """

# -----------------------------------------------------------------------------
# Modelo 1: Formato do perfil, usando apenas as espessuras
# -----------------------------------------------------------------------------

X = data[features].astype(float)
X_shape = X.sub(X.mean(axis=1), axis=0)

evaluate_kmeans(X_shape, "Formato do Perfil")

model_shape = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

data['cluster_shape'] = model_shape.fit_predict(X_shape)

profile_shape = data.groupby('cluster_shape')[features].mean()
cluster_counts_shape = data['cluster_shape'].value_counts().sort_index()

print("\nNúmero final de clusters usado - Formato do Perfil:", 3)
print("\nPerfil médio dos clusters - Formato do Perfil:")
print(profile_shape)

print("\nQuantidade de registros por cluster - Formato do Perfil:")
print(cluster_counts_shape)

# -----------------------------------------------------------------------------
# Modelo 1B: Formato do perfil + Age + Gender + Eye
# -----------------------------------------------------------------------------

X_demo = build_demographic_features(data)
X_shape_demo = pd.concat(
    [
        X_shape.reset_index(drop=True),
        X_demo.reset_index(drop=True)
    ],
    axis=1
)

evaluate_kmeans(X_shape_demo, "Formato do Perfil + Age + Gender")

model_shape_demo = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

data['cluster_shape_demo'] = model_shape_demo.fit_predict(X_shape_demo)

profile_shape_demo = data.groupby('cluster_shape_demo')[features + ["Age"]].mean()
cluster_counts_shape_demo = data['cluster_shape_demo'].value_counts().sort_index()
cluster_demographics_shape_demo = data.groupby('cluster_shape_demo')[['Gender']].agg(
    lambda s: s.value_counts(normalize=True).round(3).to_dict()
)

print("\nNúmero final de clusters usado - Formato do Perfil + Demográficos:", 3)
print("\nPerfil médio dos clusters - Formato do Perfil + Demográficos:")
print(profile_shape_demo)

print("\nDistribuição de Gender por cluster - Formato do Perfil + Demográficos:")
print(cluster_demographics_shape_demo)

print("\nQuantidade de registros por cluster - Formato do Perfil + Demográficos:")
print(cluster_counts_shape_demo)
