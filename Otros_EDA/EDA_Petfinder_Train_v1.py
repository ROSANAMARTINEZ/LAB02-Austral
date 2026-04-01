# %% [markdown]
# # EDA: PetFinder Adoption Prediction (Train Data)
# **Objetivo:** Analizar la adoptabilidad de las mascotas e identificar patrones, anomalías y el impacto del sentimiento en la velocidad de adopción (`AdoptionSpeed`).

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")

# %% [markdown]
# ## 1. Carga de Datos Tabulares
# Solo leeremos los datos de entrenamiento ubicados en la ruta especificada.

# %%
base_dir = r"C:\GIBHUB\UA_MDM_Labo2\input\petfinder-adoption-prediction"

# Buscar el csv principal (puede estar en la raíz o dentro de un subdirectorio train)
train_csv_path = os.path.join(base_dir, "train", "train.csv")
if not os.path.exists(train_csv_path):
    train_csv_path = os.path.join(base_dir, "train.csv")

df_train = pd.read_csv(train_csv_path)
print(f"Dimensiones del dataset de entrenamiento: {df_train.shape}")
display(df_train.head())

# %% [markdown]
# ## 2. Análisis de la Variable Objetivo (`AdoptionSpeed`)
# `AdoptionSpeed` va de 0 (Adoptado el mismo día) a 4 (No adoptado tras 100 días).

# %%
plt.figure(figsize=(8, 5))
sns.countplot(data=df_train, x='AdoptionSpeed', palette='viridis')
plt.title('Distribución de la Velocidad de Adopción (AdoptionSpeed)')
plt.xlabel('Velocidad de Adopción (0 = Más rápido, 4 = No adoptado)')
plt.ylabel('Cantidad de Mascotas')
plt.show()

# %% [markdown]
# ## 3. Detección de Valores Sospechosos / Deducciones Incorrectas
# Vamos a buscar outliers lógicos en los datos, como edades irreales o cantidades nulas.

# %%
# 3.1 Edades anómalas (Ej. mayores a 240 meses / 20 años)
edades_sospechosas = df_train[df_train['Age'] > 240]
print(f"Mascotas con más de 20 años registrados (sospechoso de error tipográfico o edad en días): {len(edades_sospechosas)}")

plt.figure(figsize=(10, 4))
sns.boxplot(x=df_train['Age'])
plt.title('Distribución de Edades (Meses) - Búsqueda de Outliers')
plt.show()

# 3.2 Cantidad (Quantity) de mascotas vs Fee (Tarifa)
# Si una publicación tiene una tarifa alta pero dice ser un perro "mezclado", o la cantidad es inusual.
gratis_vs_pago = df_train.copy()
gratis_vs_pago['EsGratis'] = gratis_vs_pago['Fee'] == 0

print("Porcentaje de adopciones según gratuidad y velocidad:")
display(pd.crosstab(gratis_vs_pago['EsGratis'], gratis_vs_pago['AdoptionSpeed'], normalize='index') * 100)

# 3.3 Consistencia Médica (Mascotas reportadas como sanas pero sin vacunas o con valores "Not Sure" = 3)
datos_inconsistentes = df_train[(df_train['Health'] == 1) & ((df_train['Vaccinated'] == 3) | (df_train['Dewormed'] == 3))]
print(f"Mascotas reportadas como 'Completamente Sanas' pero con estatus médico desconocido: {len(datos_inconsistentes)}")

# %% [markdown]
# ## 4. Análisis de Factores de Adoptabilidad (Bivariado)

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Tipo de mascota (1 = Perro, 2 = Gato)
sns.countplot(data=df_train, x='AdoptionSpeed', hue='Type', ax=axes[0, 0])
axes[0, 0].set_title('Velocidad de Adopción por Tipo (1=Perro, 2=Gato)')

# Género (1 = Macho, 2 = Hembra, 3 = Mixto en grupo)
sns.countplot(data=df_train, x='AdoptionSpeed', hue='Gender', ax=axes[0, 1])
axes[0, 1].set_title('Velocidad de Adopción por Género')

# Salud (1 = Sano, 2 = Herida Menor, 3 = Herida Seria)
sns.countplot(data=df_train, x='AdoptionSpeed', hue='Health', ax=axes[1, 0])
axes[1, 0].set_title('Velocidad de Adopción por Estado de Salud')

# Cantidad de Fotos vs Velocidad Media
sns.boxplot(data=df_train, x='AdoptionSpeed', y='PhotoAmt', ax=axes[1, 1])
axes[1, 1].set_title('Impacto de la Cantidad de Fotos en la Adopción')
axes[1, 1].set_ylim(0, 15) # Limitamos el eje Y para mejor visibilidad

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Extracción y Análisis de Sentimientos
# Vamos a iterar sobre la carpeta `train_sentiment`, donde cada archivo `.json` contiene el análisis de Google NLP.
# Extraeremos el `score` (positividad/negatividad) y la `magnitude` (fuerza emocional).

# %%
sentiment_dir = os.path.join(base_dir, "train_sentiment")
sentiment_data = []

if os.path.exists(sentiment_dir):
    # Listar todos los jsons
    json_files = glob.glob(os.path.join(sentiment_dir, "*.json"))
    print(f"Procesando {len(json_files)} archivos de sentimiento...")
    
    for file_path in json_files:
        pet_id = os.path.basename(file_path).split('.')[0]
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                doc_sentiment = data.get('documentSentiment', {})
                score = doc_sentiment.get('score', 0)
                magnitude = doc_sentiment.get('magnitude', 0)
                
                sentiment_data.append({
                    'PetID': pet_id,
                    'SentimentScore': score,
                    'SentimentMagnitude': magnitude
                })
            except Exception as e:
                pass
else:
    print("No se encontró la carpeta 'train_sentiment'. Revisa la ruta.")

df_sentiment = pd.DataFrame(sentiment_data)

# Unir sentimientos con el dataset principal
df_train_full = pd.merge(df_train, df_sentiment, on='PetID', how='left')

# Rellenar valores nulos para las mascotas sin descripción/sentimiento
df_train_full['SentimentScore'].fillna(0, inplace=True)
df_train_full['SentimentMagnitude'].fillna(0, inplace=True)

print("Dataset unificado con sentimientos:")
display(df_train_full[['PetID', 'AdoptionSpeed', 'SentimentScore', 'SentimentMagnitude']].head())

# %% [markdown]
# ## 6. Impacto del Sentimiento en la Adopción

# %%
plt.figure(figsize=(10, 5))
sns.violinplot(data=df_train_full, x='AdoptionSpeed', y='SentimentScore', inner="quartile", palette="coolwarm")
plt.title('Velocidad de Adopción vs Score de Sentimiento en la Descripción')
plt.xlabel('Velocidad de Adopción')
plt.ylabel('Sentiment Score (-1.0 a 1.0)')
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=df_train_full, x='AdoptionSpeed', y='SentimentMagnitude', palette="YlOrBr")
plt.title('Velocidad de Adopción vs Magnitud de Sentimiento (Emoción Acumulada)')
plt.xlabel('Velocidad de Adopción')
plt.ylabel('Sentiment Magnitude')
plt.show()

# %% [markdown]
# ## 7. Conclusiones Rápidas y Matriz de Correlación
# Verificaremos numéricamente qué variables se correlacionan más con AdoptionSpeed.

# %%
cols_numericas = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'SentimentScore', 'SentimentMagnitude', 'AdoptionSpeed']
corr_matrix = df_train_full[cols_numericas].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt='.2f', vmin=-1, vmax=1)
plt.title('Matriz de Correlación (Variables Numéricas + Sentimiento)')
plt.show()

# Conclusión: Factores como PhotoAmt y Age suelen tener más peso. El SentimentScore 
# tiende a ser más positivo en animales que se adoptan rápido, aunque la correlación 
# lineal pura suele ser baja. Modelos no lineales (como Random Forest / XGBoost) 
# capturarán mejor esta relación.
