# =============================================================================
# Exploratory Data Analysis (EDA) — PetFinder Adoption Prediction
# Dataset: PetFinder.my Adoption Prediction (Kaggle)
# Objetivo: Predecir la velocidad de adopción de mascotas en Malasia
# =============================================================================


# -----------------------------------------------------------------------------
# SECCIÓN 1: Setup e Imports
# Importación de librerías, configuración de opciones globales, definición de
# rutas, paleta de colores neón y mapeos de variables categóricas.
# -----------------------------------------------------------------------------
import os
import json
import glob
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
from scipy import stats
from IPython.display import IFrame, display, HTML

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# Rutas principales
BASE_PATH   = '/Users/fernandopaganini/Desktop/Laboratorio 2/UA_MDM_Labo2/input/petfinder-adoption-prediction'
OUTPUT_PATH = '/Users/fernandopaganini/Desktop/Laboratorio 2/output'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Tema y paleta neón para todos los gráficos
THEME = 'plotly_dark'
NEON  = [
    '#00FF9F', '#FF6B35', '#00D4FF', '#FF0099',
    '#FFE600', '#7B2FFF', '#FF3131', '#00FFFF',
    '#FF69B4', '#ADFF2F', '#FF8C00', '#39FF14'
]

# Mapeo de las etiquetas de AdoptionSpeed
ADOPTION_MAP = {
    0: '0: Mismo día',
    1: '1: 1–7 días',
    2: '2: 8–30 días',
    3: '3: 31–90 días',
    4: '4: No adoptado (+100 días)'
}
ADOPTION_ORDER = list(ADOPTION_MAP.values())

print("✅ Setup completado.")
print(f"   Output folder: {OUTPUT_PATH}")


# -----------------------------------------------------------------------------
# SECCIÓN 2: Funciones auxiliares
# top_n_others agrupa las categorías menos frecuentes en "Otros".
# save_fig y save_mpl exportan gráficos de Plotly y Matplotlib respectivamente.
# -----------------------------------------------------------------------------
def top_n_others(series, n=9, label_otros='Otros'):
    """Retorna las n categorías más frecuentes; el resto se agrupa en 'Otros'."""
    counts  = series.value_counts(dropna=False)
    top     = counts.iloc[:n]
    rest_sum = counts.iloc[n:].sum()
    if rest_sum > 0:
        top = pd.concat([top, pd.Series({label_otros: rest_sum})])
    return top

def save_fig(fig, name):
    """Guarda figura Plotly como PNG en OUTPUT_PATH."""
    fig.write_image(f'{OUTPUT_PATH}/{name}.png', width=1200, height=600)

def save_mpl(name):
    """Guarda figura Matplotlib activa como PNG en OUTPUT_PATH."""
    plt.savefig(f'{OUTPUT_PATH}/{name}.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')


# -----------------------------------------------------------------------------
# SECCIÓN 3: Carga de datos y joins con diccionarios
# Se carga el CSV principal de entrenamiento y los archivos de diccionarios
# para resolver los códigos de raza, color, estado y tipo de animal.
# Las columnas Name y Description se eliminan según las instrucciones del proyecto.
# -----------------------------------------------------------------------------
train = pd.read_csv(f'{BASE_PATH}/train/train.csv')
print(f"train.csv cargado: {train.shape[0]:,} filas × {train.shape[1]} columnas")

breed_labels = pd.read_csv(f'{BASE_PATH}/breed_labels.csv')
color_labels  = pd.read_csv(f'{BASE_PATH}/color_labels.csv')
state_labels  = pd.read_csv(f'{BASE_PATH}/state_labels.csv')

print(f"breed_labels: {len(breed_labels)} razas")
print(f"color_labels: {len(color_labels)} colores")
print(f"state_labels: {len(state_labels)} estados")

# Construcción de mapas a partir de los diccionarios
breed_dict     = dict(zip(breed_labels['BreedID'], breed_labels['BreedName']))
breed_dict[0]  = 'Mixta / No definida'
color_dict     = dict(zip(color_labels['ColorID'], color_labels['ColorName']))
color_dict[0]  = 'Sin color'
state_dict     = dict(zip(state_labels['StateID'], state_labels['StateName']))

# Aplicación de joins y creación de columnas legibles
train['TypeName']          = train['Type'].map({1: 'Perro', 2: 'Gato'})
train['Breed1Name']        = train['Breed1'].map(breed_dict).fillna('Desconocida')
train['Breed2Name']        = train['Breed2'].map(breed_dict).fillna('Sin segunda raza')
train['StateName']         = train['State'].map(state_dict).fillna('Desconocido')
train['Color1Name']        = train['Color1'].map(color_dict).fillna('Sin color')
train['Color2Name']        = train['Color2'].map(color_dict).fillna('Sin color')
train['Color3Name']        = train['Color3'].map(color_dict).fillna('Sin color')
train['AdoptionSpeedLabel'] = train['AdoptionSpeed'].map(ADOPTION_MAP)

# Mapeos de variables de salud y características físicas
yn_map     = {1: 'Sí', 2: 'No', 3: 'No sabe'}
health_map = {1: 'Saludable', 2: 'Recuperándose', 3: 'Enfermo'}
gender_map = {1: 'Macho', 2: 'Hembra', 3: 'Mixto'}
fur_map    = {1: 'Corto', 2: 'Mediano', 3: 'Largo', 0: 'Sin pelo'}
size_map   = {1: 'Pequeño', 2: 'Mediano', 3: 'Grande', 4: 'Extra grande'}

train['VaccinatedLabel']   = train['Vaccinated'].map(yn_map)
train['DewormedLabel']     = train['Dewormed'].map(yn_map)
train['SterilizedLabel']   = train['Sterilized'].map(yn_map)
train['HealthLabel']       = train['Health'].map(health_map)
train['GenderLabel']       = train['Gender'].map(gender_map)
train['FurLengthLabel']    = train['FurLength'].map(fur_map)
train['MaturitySizeLabel'] = train['MaturitySize'].map(size_map)

# Eliminación de columnas no requeridas
train = train.drop(columns=['Name', 'Description'], errors='ignore')

print(f"\nDataset enriquecido: {train.shape[0]:,} filas × {train.shape[1]} columnas")
print(train.head(3))


# -----------------------------------------------------------------------------
# SECCIÓN 4: Reporte automático con ydata-profiling
# Genera un reporte HTML de calidad de datos sobre las columnas originales
# (sin las columnas derivadas del join). El archivo se exporta a OUTPUT_PATH.
# -----------------------------------------------------------------------------
from ydata_profiling import ProfileReport

cols_originales = [c for c in train.columns
                   if not c.endswith('Name') and not c.endswith('Label')]

profile = ProfileReport(
    train[cols_originales],
    title='PetFinder Adoption — Reporte de Calidad de Datos',
    explorative=True,
    minimal=False,
    progress_bar=True
)

report_path = f'{OUTPUT_PATH}/profiling_report.html'
profile.to_file(report_path)
print(f"✅ Reporte guardado en: {report_path}")


# -----------------------------------------------------------------------------
# SECCIÓN 5: Calidad de datos — Nulos y duplicados
# Se analiza la presencia de valores faltantes (NaN reales), registros
# duplicados y estadísticas descriptivas de variables numéricas clave.
# Los ceros en VideoAmt, Fee, Color2 y Color3 se consideran valores válidos.
# -----------------------------------------------------------------------------

# Conteo y visualización de valores nulos
null_counts = train.isnull().sum()
null_pct    = (null_counts / len(train) * 100).round(2)

null_df = (
    pd.DataFrame({'Variable': null_counts.index,
                  'Nulos': null_counts.values,
                  'Porcentaje (%)': null_pct.values})
    .query('Nulos > 0')
    .sort_values('Nulos', ascending=False)
    .reset_index(drop=True)
)

print(f"Variables con valores nulos: {len(null_df)}")
print(null_df)

if len(null_df) > 0:
    fig_nulos = px.bar(
        null_df, x='Variable', y='Porcentaje (%)',
        text='Nulos',
        title='Porcentaje de Valores Nulos por Variable',
        template=THEME,
        color='Porcentaje (%)',
        color_continuous_scale='Plasma',
        labels={'Porcentaje (%)': '% de Nulos'}
    )
    fig_nulos.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig_nulos.update_layout(coloraxis_showscale=False, height=450)
    fig_nulos.show()
    save_fig(fig_nulos, 'calidad_nulos')
else:
    print("✅ No hay valores nulos en el dataset.")

# Análisis de duplicados y estadísticas descriptivas
dup_filas  = train.duplicated().sum()
dup_petids = train['PetID'].duplicated().sum()

print(f"Filas completamente duplicadas : {dup_filas:,}")
print(f"PetIDs duplicados              : {dup_petids:,}")

num_cols = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']
print("\n── Estadísticas numéricas ──")
print(train[num_cols].describe().T.round(2))

# Distribución de cantidad de fotos por mascota
fig_photoamt = px.histogram(
    train, x='PhotoAmt', nbins=40,
    title='Distribución de la Cantidad de Fotos por Mascota',
    template=THEME,
    color_discrete_sequence=[NEON[0]],
    labels={'PhotoAmt': 'Número de Fotos', 'count': 'Cantidad de Mascotas'}
)
fig_photoamt.update_layout(bargap=0.05, height=400)
fig_photoamt.show()
save_fig(fig_photoamt, 'calidad_photoamt')

# Distribución de tarifas de adopción (excluyendo gratuitas)
fig_fee = px.histogram(
    train[train['Fee'] > 0], x='Fee', nbins=40,
    title='Distribución de Tarifas de Adopción (excluyendo gratuitas)',
    template=THEME,
    color_discrete_sequence=[NEON[2]],
    labels={'Fee': 'Tarifa (MYR)', 'count': 'Cantidad'}
)
fig_fee.update_layout(bargap=0.05, height=400)
fig_fee.show()
save_fig(fig_fee, 'calidad_fee')


# -----------------------------------------------------------------------------
# SECCIÓN 6: Variable objetivo — AdoptionSpeed
# Análisis de la distribución de la variable target y su relación con
# tipo de animal, salud, vacunación, esterilización, tarifa y edad.
# -----------------------------------------------------------------------------

# Distribución global de AdoptionSpeed
speed_order  = ADOPTION_ORDER
speed_counts = train['AdoptionSpeedLabel'].value_counts().reindex(speed_order).dropna()

fig_speed_dist = px.bar(
    x=speed_counts.index, y=speed_counts.values,
    text=speed_counts.values,
    title='Distribución de la Variable Objetivo: AdoptionSpeed',
    template=THEME,
    color=speed_counts.index,
    color_discrete_sequence=NEON,
    labels={'x': 'Velocidad de Adopción', 'y': 'Cantidad de Mascotas'}
)
fig_speed_dist.update_traces(texttemplate='%{text:,}', textposition='outside')
fig_speed_dist.update_layout(showlegend=False, height=500,
    xaxis={'categoryorder': 'array', 'categoryarray': speed_order})
fig_speed_dist.show()
save_fig(fig_speed_dist, 'adoptionspeed_dist')

# AdoptionSpeed por tipo de animal (Perro vs Gato)
cross_type = (train.groupby(['AdoptionSpeedLabel', 'TypeName'])
              .size().reset_index(name='Cantidad'))

fig_speed_tipo = px.bar(
    cross_type,
    x='AdoptionSpeedLabel', y='Cantidad', color='TypeName',
    barmode='group',
    title='AdoptionSpeed por Tipo de Animal (Perro vs Gato)',
    template=THEME,
    color_discrete_sequence=[NEON[0], NEON[2]],
    labels={'AdoptionSpeedLabel': 'Velocidad', 'TypeName': 'Tipo', 'Cantidad': 'N° Mascotas'},
    category_orders={'AdoptionSpeedLabel': speed_order}
)
fig_speed_tipo.update_layout(height=480)
fig_speed_tipo.show()
save_fig(fig_speed_tipo, 'adoptionspeed_por_tipo')

# AdoptionSpeed según estado de salud
cross_health = (train.groupby(['AdoptionSpeedLabel', 'HealthLabel'])
                .size().reset_index(name='Cantidad'))

fig_speed_salud = px.bar(
    cross_health,
    x='AdoptionSpeedLabel', y='Cantidad', color='HealthLabel',
    barmode='group',
    title='AdoptionSpeed según Estado de Salud',
    template=THEME,
    color_discrete_sequence=[NEON[0], NEON[1], NEON[3]],
    labels={'AdoptionSpeedLabel': 'Velocidad', 'HealthLabel': 'Salud'},
    category_orders={'AdoptionSpeedLabel': speed_order}
)
fig_speed_salud.update_layout(height=480)
fig_speed_salud.show()
save_fig(fig_speed_salud, 'adoptionspeed_salud')

# AdoptionSpeed según estado de vacunación
cross_vac = (train.groupby(['AdoptionSpeedLabel', 'VaccinatedLabel'])
             .size().reset_index(name='Cantidad'))

fig_speed_vac = px.bar(
    cross_vac,
    x='AdoptionSpeedLabel', y='Cantidad', color='VaccinatedLabel',
    barmode='group',
    title='AdoptionSpeed según Vacunación',
    template=THEME,
    color_discrete_sequence=[NEON[0], NEON[1], NEON[4]],
    labels={'AdoptionSpeedLabel': 'Velocidad', 'VaccinatedLabel': 'Vacunado'},
    category_orders={'AdoptionSpeedLabel': speed_order}
)
fig_speed_vac.update_layout(height=480)
fig_speed_vac.show()
save_fig(fig_speed_vac, 'adoptionspeed_vacunacion')

# AdoptionSpeed según estado de esterilización
cross_ster = (train.groupby(['AdoptionSpeedLabel', 'SterilizedLabel'])
              .size().reset_index(name='Cantidad'))

fig_speed_ster = px.bar(
    cross_ster,
    x='AdoptionSpeedLabel', y='Cantidad', color='SterilizedLabel',
    barmode='group',
    title='AdoptionSpeed según Esterilización',
    template=THEME,
    color_discrete_sequence=[NEON[5], NEON[1], NEON[4]],
    labels={'AdoptionSpeedLabel': 'Velocidad', 'SterilizedLabel': 'Esterilizado'},
    category_orders={'AdoptionSpeedLabel': speed_order}
)
fig_speed_ster.update_layout(height=480)
fig_speed_ster.show()
save_fig(fig_speed_ster, 'adoptionspeed_esterilizacion')

# Distribución de tarifa según AdoptionSpeed (box plot)
fig_speed_fee = px.box(
    train, x='AdoptionSpeedLabel', y='Fee',
    title='Distribución de Tarifa según AdoptionSpeed',
    template=THEME,
    color='AdoptionSpeedLabel',
    color_discrete_sequence=NEON,
    labels={'AdoptionSpeedLabel': 'Velocidad de Adopción', 'Fee': 'Tarifa (MYR)'},
    category_orders={'AdoptionSpeedLabel': speed_order}
)
fig_speed_fee.update_layout(showlegend=False, height=480)
fig_speed_fee.show()
save_fig(fig_speed_fee, 'adoptionspeed_fee')

# Distribución de edad según AdoptionSpeed (violin plot)
fig_speed_edad = px.violin(
    train, x='AdoptionSpeedLabel', y='Age',
    title='Distribución de Edad según AdoptionSpeed',
    template=THEME,
    color='AdoptionSpeedLabel',
    color_discrete_sequence=NEON,
    box=True, points=False,
    labels={'AdoptionSpeedLabel': 'Velocidad de Adopción', 'Age': 'Edad (meses)'},
    category_orders={'AdoptionSpeedLabel': speed_order}
)
fig_speed_edad.update_layout(showlegend=False, height=480)
fig_speed_edad.show()
save_fig(fig_speed_edad, 'adoptionspeed_edad')


# -----------------------------------------------------------------------------
# SECCIÓN 7: Distribuciones principales
# Análisis de distribuciones de las variables más importantes del dataset:
# tipo de animal, razas, colores, estado, salud, género, tamaño y edad.
# -----------------------------------------------------------------------------

# Distribución por tipo de animal (pie chart)
type_counts = train['TypeName'].value_counts()

fig_tipo = px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title='Distribución por Tipo de Animal',
    template=THEME,
    color_discrete_sequence=[NEON[0], NEON[2]],
    hole=0.4
)
fig_tipo.update_traces(
    textposition='inside', textinfo='percent+label',
    hovertemplate='<b>%{label}</b><br>Cantidad: %{value:,}<br>Porcentaje: %{percent}<extra></extra>'
)
fig_tipo.update_layout(height=450)
fig_tipo.show()
save_fig(fig_tipo, 'dist_tipo')

# Top razas por tipo de animal (barras horizontales)
fig_razas_perro = None
fig_razas_gato  = None

for tipo in ['Perro', 'Gato']:
    subset    = train[train['TypeName'] == tipo]
    breed_top = top_n_others(subset['Breed1Name'], n=9)

    fig_razas = px.bar(
        x=breed_top.values, y=breed_top.index,
        orientation='h',
        text=breed_top.values,
        title=f'Top Razas de {tipo} (+ Otros)',
        template=THEME,
        color=breed_top.index,
        color_discrete_sequence=NEON,
        labels={'x': 'Cantidad', 'y': 'Raza'}
    )
    fig_razas.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig_razas.update_layout(showlegend=False, height=480,
                            yaxis={'categoryorder': 'total ascending'})
    fig_razas.show()
    save_fig(fig_razas, f'dist_razas_{tipo.lower()}')

    if tipo == 'Perro':
        fig_razas_perro = fig_razas
    else:
        fig_razas_gato = fig_razas

# Top razas diferenciadas por color principal (barras apiladas)
fig_color_perro = None
fig_color_gato  = None

for tipo in ['Perro', 'Gato']:
    subset     = train[train['TypeName'] == tipo]
    top_breeds = subset['Breed1Name'].value_counts().head(8).index.tolist()
    sub_top    = subset[subset['Breed1Name'].isin(top_breeds)]

    cross = (sub_top.groupby(['Breed1Name', 'Color1Name'])
             .size().reset_index(name='Cantidad'))

    fig_rc = px.bar(
        cross, x='Breed1Name', y='Cantidad', color='Color1Name',
        barmode='stack',
        title=f'Top Razas de {tipo} por Color Principal',
        template=THEME,
        color_discrete_sequence=NEON,
        labels={'Breed1Name': 'Raza', 'Color1Name': 'Color', 'Cantidad': 'N°'}
    )
    fig_rc.update_layout(height=500, xaxis_tickangle=-35)
    fig_rc.show()
    save_fig(fig_rc, f'razas_color_{tipo.lower()}')

    if tipo == 'Perro':
        fig_color_perro = fig_rc
    else:
        fig_color_gato = fig_rc

# Variables de salud: Vaccinated, Dewormed, Sterilized, Health (pie charts 2x2)
health_vars = [
    ('VaccinatedLabel',  'Vacunado'),
    ('DewormedLabel',    'Desparasitado'),
    ('SterilizedLabel',  'Esterilizado'),
    ('HealthLabel',      'Estado de Salud')
]

fig_salud = make_subplots(
    rows=2, cols=2,
    subplot_titles=[v[1] for v in health_vars],
    specs=[[{'type': 'pie'}, {'type': 'pie'}],
           [{'type': 'pie'}, {'type': 'pie'}]]
)

for idx, (col, label) in enumerate(health_vars):
    row     = idx // 2 + 1
    col_pos = idx % 2 + 1
    counts  = train[col].value_counts()
    fig_salud.add_trace(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            name=label,
            marker_colors=NEON,
            textinfo='percent+label',
            hole=0.35
        ),
        row=row, col=col_pos
    )

fig_salud.update_layout(
    title_text='Distribución de Variables de Salud',
    template=THEME,
    height=700,
    showlegend=False
)
fig_salud.show()
save_fig(fig_salud, 'dist_salud')

# Distribución de género
gender_counts = train['GenderLabel'].value_counts()

fig_genero = px.pie(
    values=gender_counts.values, names=gender_counts.index,
    title='Distribución por Género',
    template=THEME,
    color_discrete_sequence=[NEON[2], NEON[3], NEON[4]],
    hole=0.4
)
fig_genero.update_traces(textposition='inside', textinfo='percent+label')
fig_genero.update_layout(height=430)
fig_genero.show()
save_fig(fig_genero, 'dist_genero')

# Distribución por estado de Malasia (barras horizontales)
state_counts = top_n_others(train['StateName'], n=9)

fig_estado = px.bar(
    x=state_counts.values, y=state_counts.index,
    orientation='h',
    text=state_counts.values,
    title='Distribución por Estado de Malasia',
    template=THEME,
    color=state_counts.index,
    color_discrete_sequence=NEON,
    labels={'x': 'Cantidad', 'y': 'Estado'}
)
fig_estado.update_traces(texttemplate='%{text:,}', textposition='outside')
fig_estado.update_layout(showlegend=False, height=450,
                         yaxis={'categoryorder': 'total ascending'})
fig_estado.show()
save_fig(fig_estado, 'dist_estado')

# Distribución de color principal (Color1)
color1_counts = train['Color1Name'].value_counts()

fig_color1 = px.bar(
    x=color1_counts.index, y=color1_counts.values,
    text=color1_counts.values,
    title='Distribución de Color Principal (Color1)',
    template=THEME,
    color=color1_counts.index,
    color_discrete_sequence=NEON,
    labels={'x': 'Color', 'y': 'Cantidad'}
)
fig_color1.update_traces(texttemplate='%{text:,}', textposition='outside')
fig_color1.update_layout(showlegend=False, height=430)
fig_color1.show()
save_fig(fig_color1, 'dist_color1')

# Distribución de tamaño de madurez y largo del pelo (pie charts 1x2)
fig_tamano_pelo = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Tamaño de Madurez', 'Largo del Pelo'],
    specs=[[{'type': 'pie'}, {'type': 'pie'}]]
)

size_c = train['MaturitySizeLabel'].value_counts()
fur_c  = train['FurLengthLabel'].value_counts()

fig_tamano_pelo.add_trace(
    go.Pie(labels=size_c.index, values=size_c.values,
           marker_colors=NEON, hole=0.35, textinfo='percent+label'),
    row=1, col=1
)
fig_tamano_pelo.add_trace(
    go.Pie(labels=fur_c.index, values=fur_c.values,
           marker_colors=NEON[4:], hole=0.35, textinfo='percent+label'),
    row=1, col=2
)

fig_tamano_pelo.update_layout(
    template=THEME, height=420, showlegend=False,
    title_text='Tamaño de Madurez y Largo del Pelo'
)
fig_tamano_pelo.show()
save_fig(fig_tamano_pelo, 'dist_tamano_pelo')

# Distribución de edad (meses) por tipo de animal
fig_edad_tipo = px.histogram(
    train, x='Age', color='TypeName', nbins=50,
    barmode='overlay',
    title='Distribución de Edad por Tipo de Animal',
    template=THEME,
    color_discrete_sequence=[NEON[0], NEON[2]],
    labels={'Age': 'Edad (meses)', 'count': 'Cantidad', 'TypeName': 'Tipo'},
    opacity=0.75
)
fig_edad_tipo.update_layout(height=450)
fig_edad_tipo.show()
save_fig(fig_edad_tipo, 'dist_edad_tipo')


# -----------------------------------------------------------------------------
# SECCIÓN 8: Metadata de imágenes (train_metadata)
# Se procesan los archivos JSON generados por Google Cloud Vision API.
# Se extraen etiquetas (labelAnnotations), colores dominantes ponderados por
# pixelFraction, cropHints y presencia de faceAnnotations / textAnnotations.
# -----------------------------------------------------------------------------
meta_files = glob.glob(f'{BASE_PATH}/train_metadata/*.json')
print(f"Total archivos de metadata: {len(meta_files):,}")

meta_records  = []
label_records = []

for filepath in meta_files:
    filename = os.path.basename(filepath)
    stem     = filename.replace('.json', '')
    parts    = stem.rsplit('-', 1)
    pet_id   = parts[0]
    img_num  = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        continue

    # Extracción de etiquetas (labels)
    for lbl in data.get('labelAnnotations', []):
        label_records.append({
            'PetID':       pet_id,
            'ImageNum':    img_num,
            'description': lbl.get('description', ''),
            'score':       lbl.get('score', 0),
            'topicality':  lbl.get('topicality', 0)
        })

    # Cálculo del color dominante ponderado por pixelFraction
    colors = (data.get('imagePropertiesAnnotation', {})
                  .get('dominantColors', {})
                  .get('colors', []))
    if colors:
        total_frac = sum(c.get('pixelFraction', 0) for c in colors)
        if total_frac > 0:
            dom_r = sum(c.get('color', {}).get('red',   0) * c.get('pixelFraction', 0) for c in colors) / total_frac
            dom_g = sum(c.get('color', {}).get('green', 0) * c.get('pixelFraction', 0) for c in colors) / total_frac
            dom_b = sum(c.get('color', {}).get('blue',  0) * c.get('pixelFraction', 0) for c in colors) / total_frac
        else:
            dom_r = dom_g = dom_b = 128
        max_frac = max(c.get('pixelFraction', 0) for c in colors)
    else:
        dom_r = dom_g = dom_b = 128
        max_frac = 0

    # Confianza de cropHints
    crop_hints = data.get('cropHintsAnnotation', {}).get('cropHints', [])
    crop_conf  = max((h.get('confidence', 0) for h in crop_hints), default=0)

    # Presencia de faceAnnotations y textAnnotations
    try:
        has_face = len(data.get('faceAnnotations', [])) > 0
    except Exception:
        has_face = False

    try:
        has_text = len(data.get('textAnnotations', [])) > 0
    except Exception:
        has_text = False

    meta_records.append({
        'PetID':           pet_id,
        'ImageNum':        img_num,
        'dom_R':           dom_r,
        'dom_G':           dom_g,
        'dom_B':           dom_b,
        'max_pixel_frac':  max_frac,
        'crop_confidence': crop_conf,
        'has_face':        has_face,
        'has_text':        has_text,
        'n_labels':        len(data.get('labelAnnotations', []))
    })

meta_df   = pd.DataFrame(meta_records)
labels_df = pd.DataFrame(label_records)

print(f"\n✅ Metadata procesada: {len(meta_df):,} imágenes")
print(f"   Etiquetas totales:   {len(labels_df):,}")
print(f"   PetIDs únicos:       {meta_df['PetID'].nunique():,}")

# Top 20 etiquetas más frecuentes detectadas por Vision API
label_freq = (labels_df.groupby('description')
              .agg(frecuencia=('PetID', 'count'),
                   score_prom=('score', 'mean'),
                   topicality_prom=('topicality', 'mean'))
              .sort_values('frecuencia', ascending=False)
              .head(20)
              .reset_index())

fig_meta_labels = px.bar(
    label_freq.sort_values('frecuencia'),
    x='frecuencia', y='description',
    orientation='h',
    text='frecuencia',
    title='Top 20 Etiquetas más Frecuentes en Imágenes (Google Vision)',
    template=THEME,
    color='frecuencia',
    color_continuous_scale='Viridis',
    labels={'frecuencia': 'Frecuencia', 'description': 'Etiqueta'}
)
fig_meta_labels.update_traces(texttemplate='%{text:,}', textposition='outside')
fig_meta_labels.update_layout(coloraxis_showscale=False, height=600)
fig_meta_labels.show()
save_fig(fig_meta_labels, 'meta_top20_labels')

# Comparación score vs topicality para los top 15 labels
top15_labels = label_freq.head(15).copy()

fig_score_topic = go.Figure()
fig_score_topic.add_trace(go.Bar(
    name='Score Promedio',
    x=top15_labels['description'],
    y=top15_labels['score_prom'],
    marker_color=NEON[0],
    hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>'
))
fig_score_topic.add_trace(go.Bar(
    name='Topicality Promedio',
    x=top15_labels['description'],
    y=top15_labels['topicality_prom'],
    marker_color=NEON[2],
    hovertemplate='<b>%{x}</b><br>Topicality: %{y:.3f}<extra></extra>'
))
fig_score_topic.update_layout(
    barmode='group',
    title='Comparación Score vs Topicality — Top 15 Etiquetas',
    template=THEME,
    xaxis_tickangle=-40,
    height=480,
    yaxis_title='Valor Promedio',
    xaxis_title='Etiqueta'
)
fig_score_topic.show()
save_fig(fig_score_topic, 'meta_score_vs_topicality')

# Distribución de los canales RGB del color dominante ponderado
avg_r = meta_df['dom_R'].mean()
avg_g = meta_df['dom_G'].mean()
avg_b = meta_df['dom_B'].mean()
avg_color_hex = '#{:02x}{:02x}{:02x}'.format(int(avg_r), int(avg_g), int(avg_b))
print(f"Color promedio ponderado del dataset: RGB({avg_r:.0f}, {avg_g:.0f}, {avg_b:.0f}) → {avg_color_hex}")

fig_rgb = go.Figure()
for canal, col_name, color in zip(['dom_R', 'dom_G', 'dom_B'],
                                   ['Rojo', 'Verde', 'Azul'],
                                   ['#FF4444', '#44FF44', '#4444FF']):
    fig_rgb.add_trace(go.Histogram(
        x=meta_df[canal], name=canal, nbinsx=50,
        marker_color=color, opacity=0.7,
        hovertemplate=f'<b>{canal}</b><br>Valor: %{{x}}<br>N°: %{{y}}<extra></extra>'
    ))

fig_rgb.update_layout(
    barmode='overlay',
    title='Distribución de los Canales RGB del Color Dominante Ponderado',
    template=THEME,
    xaxis_title='Valor del Canal (0–255)',
    yaxis_title='N° de Imágenes',
    height=450
)
fig_rgb.show()
save_fig(fig_rgb, 'meta_rgb_dist')

# Paleta de colores dominantes del dataset (muestra de 200 imágenes con Matplotlib)
fig_mpl, ax = plt.subplots(1, 1, figsize=(10, 2), facecolor='#1a1a2e')

sample = meta_df.sample(min(200, len(meta_df)), random_state=42).reset_index(drop=True)
for i, row in sample.iterrows():
    r, g, b = int(row['dom_R']), int(row['dom_G']), int(row['dom_B'])
    rect = mpatches.Rectangle((i / len(sample), 0), 1 / len(sample), 1,
                               color=(r/255, g/255, b/255))
    ax.add_patch(rect)

ax.add_patch(mpatches.Rectangle((0, -0.4), 1, 0.35,
                                  color=(avg_r/255, avg_g/255, avg_b/255)))
ax.text(0.5, -0.23, f'Color Promedio: {avg_color_hex}',
        ha='center', va='center', fontsize=11,
        color='white', transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(-0.45, 1)
ax.axis('off')
ax.set_title('Paleta de Colores Dominantes del Dataset (muestra de 200 imágenes)',
             color='white', pad=10, fontsize=13)
plt.tight_layout()
plt.show()
save_mpl('meta_paleta_colores')
plt.close()

# Distribución de pixelFraction del color dominante por imagen
fig_pixfrac = px.histogram(
    meta_df, x='max_pixel_frac', nbins=40,
    title='Distribución de pixelFraction del Color Dominante por Imagen',
    template=THEME,
    color_discrete_sequence=[NEON[4]],
    labels={'max_pixel_frac': 'Fracción de Píxeles del Color Dominante',
            'count': 'N° de Imágenes'}
)
fig_pixfrac.add_vline(x=meta_df['max_pixel_frac'].median(), line_dash='dash',
                      line_color=NEON[0],
                      annotation_text=f"Mediana: {meta_df['max_pixel_frac'].median():.2f}",
                      annotation_position='top right')
fig_pixfrac.update_layout(height=430)
fig_pixfrac.show()
save_fig(fig_pixfrac, 'meta_pixel_fraction')

# Porcentaje de imágenes con cropHints de alta confianza (> 0.8)
high_conf = (meta_df['crop_confidence'] > 0.8).mean() * 100
low_conf  = 100 - high_conf

fig_crophints = px.pie(
    values=[high_conf, low_conf],
    names=['Alta confianza (>0.8)', 'Baja confianza (≤0.8)'],
    title='% Imágenes con CropHints de Alta Confianza',
    template=THEME,
    color_discrete_sequence=[NEON[0], NEON[3]],
    hole=0.4
)
fig_crophints.update_traces(textinfo='percent+label')
fig_crophints.update_layout(height=430)
fig_crophints.show()
save_fig(fig_crophints, 'meta_crophints_conf')
print(f"Imágenes con crop confidence > 0.8: {high_conf:.1f}%")

# Presencia de anotaciones especiales (rostros y texto) en las imágenes
pct_face = meta_df['has_face'].mean() * 100
pct_text = meta_df['has_text'].mean() * 100
pct_none = 100 - pct_face - pct_text

print(f"% imágenes con faceAnnotations : {pct_face:.2f}%")
print(f"% imágenes con textAnnotations : {pct_text:.2f}%")

fig_anotaciones = px.bar(
    x=['Con Rostros Detectados', 'Con Texto Detectado', 'Sin anotaciones especiales'],
    y=[pct_face, pct_text, max(0, pct_none)],
    text=[f'{pct_face:.1f}%', f'{pct_text:.1f}%', f'{max(0,pct_none):.1f}%'],
    title='Presencia de Anotaciones Especiales en Imágenes',
    template=THEME,
    color=['Con Rostros Detectados', 'Con Texto Detectado', 'Sin anotaciones especiales'],
    color_discrete_sequence=[NEON[3], NEON[2], NEON[6]],
    labels={'x': 'Tipo de Anotación', 'y': 'Porcentaje (%)'}
)
fig_anotaciones.update_traces(textposition='outside')
fig_anotaciones.update_layout(showlegend=False, height=430)
fig_anotaciones.show()
save_fig(fig_anotaciones, 'meta_anotaciones_especiales')

# WordCloud de etiquetas ponderadas por score promedio (Vision API)
label_scores = (labels_df.groupby('description')['score']
                .mean()
                .to_dict())

wc = WordCloud(
    width=1200, height=500,
    background_color='black',
    colormap='cool',
    max_words=120,
    collocations=False
).generate_from_frequencies(label_scores)

fig_mpl, ax = plt.subplots(figsize=(14, 6), facecolor='black')
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_title('WordCloud de Etiquetas Ponderadas por Score Promedio (Google Vision)',
             color='#00FF9F', fontsize=14, pad=12)
plt.tight_layout()
plt.show()
save_mpl('meta_wordcloud_labels')
plt.close()


# -----------------------------------------------------------------------------
# SECCIÓN 9: Análisis de sentimientos (train_sentiment)
# Se procesan los archivos JSON de la Google Cloud Natural Language API.
# Se extraen: score y magnitude del documento, variación entre oraciones,
# entidades con salience, idioma detectado. Todo se cruza con AdoptionSpeed.
# -----------------------------------------------------------------------------
sent_files = glob.glob(f'{BASE_PATH}/train_sentiment/*.json')
print(f"Total archivos de sentimientos: {len(sent_files):,}")

sent_records   = []
entity_records = []

for filepath in sent_files:
    pet_id = os.path.basename(filepath).replace('.json', '')

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        continue

    # Sentimiento global del documento
    doc_sent = data.get('documentSentiment', {})
    g_score  = doc_sent.get('score',     np.nan)
    g_mag    = doc_sent.get('magnitude', np.nan)

    # Métricas a nivel de oraciones
    sentences   = data.get('sentences', [])
    n_sentences = len(sentences)
    sent_scores = [s.get('sentiment', {}).get('score', 0) for s in sentences]
    sent_var    = (max(sent_scores) - min(sent_scores)) if len(sent_scores) > 1 else 0.0

    sent_records.append({
        'PetID':          pet_id,
        'global_score':   g_score,
        'global_mag':     g_mag,
        'n_sentences':    n_sentences,
        'sent_variation': sent_var,
        'language':       data.get('language', 'desconocido')
    })

    # Extracción de entidades con nombre, tipo y salience
    for ent in data.get('entities', []):
        entity_records.append({
            'PetID':    pet_id,
            'name':     ent.get('name', ''),
            'type':     ent.get('type', 'OTHER'),
            'salience': ent.get('salience', 0)
        })

sent_df   = pd.DataFrame(sent_records)
entity_df = pd.DataFrame(entity_records)

# Join con train para incorporar AdoptionSpeed y TypeName
sent_df = sent_df.merge(
    train[['PetID', 'AdoptionSpeed', 'AdoptionSpeedLabel', 'TypeName']],
    on='PetID', how='left'
)

# Clasificación del sentimiento global en tres categorías
def classify_sent(score):
    if pd.isna(score):
        return 'Neutro'
    if score > 0.25:
        return 'Positivo'
    elif score < -0.25:
        return 'Negativo'
    else:
        return 'Neutro'

sent_df['SentCategory'] = sent_df['global_score'].apply(classify_sent)

print(f"\n✅ Sentimientos procesados: {len(sent_df):,} registros")
print(f"   Entidades procesadas:     {len(entity_df):,}")
print(sent_df.head(3))

# Distribución del score global por tipo de animal
fig_sent_score = px.histogram(
    sent_df.dropna(subset=['global_score']),
    x='global_score',
    color='TypeName',
    nbins=40,
    barmode='overlay',
    opacity=0.75,
    title='Distribución del Score de Sentimiento por Tipo de Animal',
    template=THEME,
    color_discrete_sequence=[NEON[0], NEON[2]],
    labels={'global_score': 'Score de Sentimiento (-1 a +1)',
            'count': 'N° Mascotas', 'TypeName': 'Tipo'}
)
fig_sent_score.add_vline(x=0,     line_dash='dash', line_color='white', opacity=0.5)
fig_sent_score.add_vline(x=0.25,  line_dash='dot',  line_color=NEON[3], opacity=0.7,
                         annotation_text='Umbral Positivo')
fig_sent_score.add_vline(x=-0.25, line_dash='dot',  line_color=NEON[6], opacity=0.7,
                         annotation_text='Umbral Negativo')
fig_sent_score.update_layout(height=470)
fig_sent_score.show()
save_fig(fig_sent_score, 'sent_score_tipo')

# Distribución de la magnitude global del sentimiento
fig_sent_mag = px.histogram(
    sent_df.dropna(subset=['global_mag']),
    x='global_mag', nbins=40,
    title='Distribución de la Magnitude Global del Sentimiento',
    template=THEME,
    color_discrete_sequence=[NEON[5]],
    labels={'global_mag': 'Magnitude (intensidad emocional)',
            'count': 'N° Mascotas'}
)
fig_sent_mag.add_vline(x=sent_df['global_mag'].median(), line_dash='dash',
                       line_color=NEON[0],
                       annotation_text=f"Mediana: {sent_df['global_mag'].median():.2f}")
fig_sent_mag.update_layout(height=440)
fig_sent_mag.show()
save_fig(fig_sent_mag, 'sent_magnitude')

# Correlación de Spearman entre score de sentimiento y AdoptionSpeed
corr_data = sent_df.dropna(subset=['global_score', 'AdoptionSpeed'])
rho, p_val = stats.spearmanr(corr_data['global_score'], corr_data['AdoptionSpeed'])
print(f"Correlación de Spearman (score vs AdoptionSpeed): ρ = {rho:.4f}, p = {p_val:.4f}")

fig_sent_adoption = px.box(
    sent_df.dropna(subset=['global_score', 'AdoptionSpeedLabel']),
    x='AdoptionSpeedLabel', y='global_score',
    title=f'Score de Sentimiento por AdoptionSpeed  (ρ = {rho:.3f})',
    template=THEME,
    color='AdoptionSpeedLabel',
    color_discrete_sequence=NEON,
    points='outliers',
    labels={'AdoptionSpeedLabel': 'Velocidad de Adopción',
            'global_score': 'Score Sentimiento'},
    category_orders={'AdoptionSpeedLabel': ADOPTION_ORDER}
)
fig_sent_adoption.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
fig_sent_adoption.update_layout(showlegend=False, height=490)
fig_sent_adoption.show()
save_fig(fig_sent_adoption, 'sent_score_adoptionspeed')

# Clasificación de publicaciones en positivo / negativo / neutro
cat_counts = sent_df['SentCategory'].value_counts()

fig_sent_cat = px.pie(
    values=cat_counts.values, names=cat_counts.index,
    title='Clasificación de Publicaciones por Sentimiento',
    template=THEME,
    color=cat_counts.index,
    color_discrete_map={'Positivo': NEON[0], 'Negativo': NEON[6], 'Neutro': NEON[4]},
    hole=0.4
)
fig_sent_cat.update_traces(textinfo='percent+label')
fig_sent_cat.update_layout(height=440)
fig_sent_cat.show()
save_fig(fig_sent_cat, 'sent_clasificacion')

# Cruce categoría de sentimiento vs AdoptionSpeed
cross_cat = (sent_df.dropna(subset=['AdoptionSpeedLabel'])
             .groupby(['AdoptionSpeedLabel', 'SentCategory'])
             .size().reset_index(name='Cantidad'))

fig_sent_cat_speed = px.bar(
    cross_cat,
    x='AdoptionSpeedLabel', y='Cantidad', color='SentCategory',
    barmode='group',
    title='AdoptionSpeed por Categoría de Sentimiento',
    template=THEME,
    color_discrete_map={'Positivo': NEON[0], 'Negativo': NEON[6], 'Neutro': NEON[4]},
    labels={'AdoptionSpeedLabel': 'Velocidad', 'SentCategory': 'Sentimiento'},
    category_orders={'AdoptionSpeedLabel': ADOPTION_ORDER,
                     'SentCategory': ['Positivo', 'Neutro', 'Negativo']}
)
fig_sent_cat_speed.update_layout(height=480)
fig_sent_cat_speed.show()
save_fig(fig_sent_cat_speed, 'sent_categoria_adoptionspeed')

# Variación interna del sentimiento (max–min entre oraciones) vs AdoptionSpeed
fig_sent_var = px.box(
    sent_df.dropna(subset=['sent_variation', 'AdoptionSpeedLabel']),
    x='AdoptionSpeedLabel', y='sent_variation',
    title='Variación Interna del Sentimiento por AdoptionSpeed\n(diferencia max–min entre oraciones)',
    template=THEME,
    color='AdoptionSpeedLabel',
    color_discrete_sequence=NEON,
    points='outliers',
    labels={'AdoptionSpeedLabel': 'Velocidad', 'sent_variation': 'Variación (max–min)'},
    category_orders={'AdoptionSpeedLabel': ADOPTION_ORDER}
)
fig_sent_var.update_layout(showlegend=False, height=490)
fig_sent_var.show()
save_fig(fig_sent_var, 'sent_variacion_adoptionspeed')

v_corr, v_p = stats.spearmanr(
    sent_df.dropna(subset=['sent_variation', 'AdoptionSpeed'])['sent_variation'],
    sent_df.dropna(subset=['sent_variation', 'AdoptionSpeed'])['AdoptionSpeed']
)
print(f"Correlación de Spearman (variación vs AdoptionSpeed): ρ = {v_corr:.4f}, p = {v_p:.4f}")

# Entidades más frecuentes por tipo (PERSON, LOCATION, OTHER)
fig_ent_person   = None
fig_ent_location = None
fig_ent_other    = None

for etype in ['PERSON', 'LOCATION', 'OTHER']:
    sub     = entity_df[entity_df['type'] == etype]
    ent_top = top_n_others(sub['name'], n=9)

    fig_ent = px.bar(
        x=ent_top.values, y=ent_top.index,
        orientation='h',
        text=ent_top.values,
        title=f'Entidades más Frecuentes — Tipo: {etype}',
        template=THEME,
        color=ent_top.index,
        color_discrete_sequence=NEON,
        labels={'x': 'Frecuencia', 'y': 'Entidad'}
    )
    fig_ent.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig_ent.update_layout(showlegend=False, height=450,
                          yaxis={'categoryorder': 'total ascending'})
    fig_ent.show()
    save_fig(fig_ent, f'sent_entidades_{etype.lower()}')

    if etype == 'PERSON':
        fig_ent_person = fig_ent
    elif etype == 'LOCATION':
        fig_ent_location = fig_ent
    else:
        fig_ent_other = fig_ent

# WordCloud de entidades ponderadas por salience promedio
entity_salience          = entity_df.groupby('name')['salience'].mean().to_dict()
entity_salience_filtered = {k: v for k, v in entity_salience.items() if len(k) > 2}

wc_ent = WordCloud(
    width=1200, height=500,
    background_color='black',
    colormap='spring',
    max_words=100,
    collocations=False
).generate_from_frequencies(entity_salience_filtered)

fig_mpl, ax = plt.subplots(figsize=(14, 6), facecolor='black')
ax.imshow(wc_ent, interpolation='bilinear')
ax.axis('off')
ax.set_title('WordCloud de Entidades Ponderadas por Salience Promedio',
             color='#FF6B35', fontsize=14, pad=12)
plt.tight_layout()
plt.show()
save_mpl('sent_wordcloud_entidades')
plt.close()

# Cantidad de oraciones en la descripción vs AdoptionSpeed
fig_sent_oraciones = px.box(
    sent_df.dropna(subset=['AdoptionSpeedLabel']),
    x='AdoptionSpeedLabel', y='n_sentences',
    title='Número de Oraciones en la Descripción por AdoptionSpeed',
    template=THEME,
    color='AdoptionSpeedLabel',
    color_discrete_sequence=NEON,
    points='outliers',
    labels={'AdoptionSpeedLabel': 'Velocidad', 'n_sentences': 'N° Oraciones'},
    category_orders={'AdoptionSpeedLabel': ADOPTION_ORDER}
)
fig_sent_oraciones.update_layout(showlegend=False, height=480)
fig_sent_oraciones.show()
save_fig(fig_sent_oraciones, 'sent_oraciones_adoptionspeed')

n_corr, n_p = stats.spearmanr(
    sent_df.dropna(subset=['n_sentences', 'AdoptionSpeed'])['n_sentences'],
    sent_df.dropna(subset=['n_sentences', 'AdoptionSpeed'])['AdoptionSpeed']
)
print(f"Correlación de Spearman (n_sentences vs AdoptionSpeed): ρ = {n_corr:.4f}, p = {n_p:.4f}")

# Heatmap de correlaciones de Spearman entre variables de sentimiento y AdoptionSpeed
num_cols_sent = ['global_score', 'global_mag', 'n_sentences', 'sent_variation', 'AdoptionSpeed']
sent_corr = sent_df[num_cols_sent].dropna().corr(method='spearman').round(3)

fig_sent_corr = px.imshow(
    sent_corr,
    text_auto=True,
    title='Mapa de Correlaciones de Spearman — Variables de Sentimiento',
    template=THEME,
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1,
    aspect='auto'
)
fig_sent_corr.update_layout(height=450)
fig_sent_corr.show()
save_fig(fig_sent_corr, 'sent_corr_heatmap')


# =============================================================================
# SECCIÓN 10: Dashboard interactivo en HTML (plotly.io)
# Se ensamblan todos los gráficos de Plotly generados en las secciones
# anteriores en un único archivo HTML con navegación por secciones,
# usando plotly.io.to_html para embeber cada figura como div independiente.
# =============================================================================
import plotly.io as pio

# Mapa de todas las figuras con su sección y título para el dashboard
dashboard_sections = [
    {
        'id':     'sec-calidad',
        'titulo': '📊 Calidad de Datos',
        'figs': [
            (fig_nulos if len(null_df) > 0 else None, 'Valores Nulos por Variable'),
            (fig_photoamt,                             'Distribución de Fotos por Mascota'),
            (fig_fee,                                  'Distribución de Tarifas de Adopción'),
        ]
    },
    {
        'id':     'sec-target',
        'titulo': '🎯 Variable Objetivo: AdoptionSpeed',
        'figs': [
            (fig_speed_dist,  'Distribución Global de AdoptionSpeed'),
            (fig_speed_tipo,  'AdoptionSpeed por Tipo de Animal'),
            (fig_speed_salud, 'AdoptionSpeed por Estado de Salud'),
            (fig_speed_vac,   'AdoptionSpeed por Vacunación'),
            (fig_speed_ster,  'AdoptionSpeed por Esterilización'),
            (fig_speed_fee,   'Tarifa vs AdoptionSpeed'),
            (fig_speed_edad,  'Edad vs AdoptionSpeed'),
        ]
    },
    {
        'id':     'sec-distribuciones',
        'titulo': '📈 Distribuciones Principales',
        'figs': [
            (fig_tipo,         'Distribución por Tipo de Animal'),
            (fig_razas_perro,  'Top Razas — Perro'),
            (fig_razas_gato,   'Top Razas — Gato'),
            (fig_color_perro,  'Razas de Perro por Color'),
            (fig_color_gato,   'Razas de Gato por Color'),
            (fig_salud,        'Variables de Salud'),
            (fig_genero,       'Distribución por Género'),
            (fig_estado,       'Distribución por Estado de Malasia'),
            (fig_color1,       'Distribución de Color Principal'),
            (fig_tamano_pelo,  'Tamaño de Madurez y Largo del Pelo'),
            (fig_edad_tipo,    'Distribución de Edad por Tipo'),
        ]
    },
    {
        'id':     'sec-metadata',
        'titulo': '🖼️ Metadata de Imágenes',
        'figs': [
            (fig_meta_labels,  'Top 20 Etiquetas (Google Vision)'),
            (fig_score_topic,  'Score vs Topicality — Top 15 Etiquetas'),
            (fig_rgb,          'Distribución RGB del Color Dominante'),
            (fig_pixfrac,      'pixelFraction del Color Dominante'),
            (fig_crophints,    'CropHints de Alta Confianza'),
            (fig_anotaciones,  'Anotaciones Especiales en Imágenes'),
        ]
    },
    {
        'id':     'sec-sentimiento',
        'titulo': '💬 Análisis de Sentimientos',
        'figs': [
            (fig_sent_score,      'Score de Sentimiento por Tipo'),
            (fig_sent_mag,        'Magnitude Global del Sentimiento'),
            (fig_sent_adoption,   'Score de Sentimiento vs AdoptionSpeed'),
            (fig_sent_cat,        'Clasificación por Sentimiento'),
            (fig_sent_cat_speed,  'Sentimiento vs AdoptionSpeed'),
            (fig_sent_var,        'Variación Interna del Sentimiento'),
            (fig_ent_person,      'Entidades — PERSON'),
            (fig_ent_location,    'Entidades — LOCATION'),
            (fig_ent_other,       'Entidades — OTHER'),
            (fig_sent_oraciones,  'Oraciones vs AdoptionSpeed'),
            (fig_sent_corr,       'Mapa de Correlaciones de Spearman'),
        ]
    },
]

# Construcción del HTML del dashboard con navegación lateral
def build_dashboard_html(sections):
    """
    Genera el HTML completo del dashboard a partir de la lista de secciones.
    Cada figura se embebe como div independiente usando plotly.io.to_html.
    """

    # Estilos CSS del dashboard
    css = """
    <style>
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body {
        font-family: 'Segoe UI', sans-serif;
        background: #0d0d1a;
        color: #e0e0e0;
        display: flex;
        min-height: 100vh;
      }

      /* ── Barra lateral ── */
      #sidebar {
        width: 240px;
        min-width: 240px;
        background: #12122a;
        border-right: 1px solid #2a2a4a;
        padding: 24px 0;
        position: sticky;
        top: 0;
        height: 100vh;
        overflow-y: auto;
      }
      #sidebar h2 {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #00FF9F;
        padding: 0 20px 16px;
        border-bottom: 1px solid #2a2a4a;
        margin-bottom: 12px;
      }
      #sidebar a {
        display: block;
        padding: 10px 20px;
        font-size: 13px;
        color: #aaa;
        text-decoration: none;
        border-left: 3px solid transparent;
        transition: all 0.2s;
      }
      #sidebar a:hover,
      #sidebar a.active {
        color: #00FF9F;
        border-left-color: #00FF9F;
        background: rgba(0,255,159,0.07);
      }

      /* ── Contenido principal ── */
      #main {
        flex: 1;
        padding: 32px 40px;
        overflow-y: auto;
      }
      #main h1 {
        font-size: 26px;
        font-weight: 800;
        color: #00FF9F;
        margin-bottom: 6px;
      }
      #main .subtitle {
        font-size: 13px;
        color: #666;
        margin-bottom: 36px;
        padding-bottom: 16px;
        border-bottom: 1px solid #2a2a4a;
      }

      /* ── Secciones ── */
      .section {
        margin-bottom: 60px;
        scroll-margin-top: 24px;
      }
      .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #00D4FF;
        padding: 10px 0 6px;
        border-bottom: 2px solid #00D4FF33;
        margin-bottom: 24px;
      }

      /* ── Grid de gráficos ── */
      .charts-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
        gap: 24px;
      }
      .chart-card {
        background: #12122a;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        overflow: hidden;
        transition: box-shadow 0.2s;
      }
      .chart-card:hover {
        box-shadow: 0 0 18px rgba(0,255,159,0.18);
      }
      .chart-card-title {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #888;
        padding: 12px 16px 0;
      }
      .chart-card .plotly-graph-div {
        width: 100% !important;
      }
    </style>
    """

    # JavaScript para resaltar sección activa en sidebar al hacer scroll
    js = """
    <script>
      document.addEventListener('DOMContentLoaded', () => {
        const links = document.querySelectorAll('#sidebar a');
        const observer = new IntersectionObserver(entries => {
          entries.forEach(e => {
            if (e.isIntersecting) {
              links.forEach(l => l.classList.remove('active'));
              const link = document.querySelector(`#sidebar a[href="#${e.target.id}"]`);
              if (link) link.classList.add('active');
            }
          });
        }, { threshold: 0.15 });
        document.querySelectorAll('.section').forEach(s => observer.observe(s));
      });
    </script>
    """

    # Construcción del sidebar con los enlaces a cada sección
    sidebar_links = '\n'.join(
        f'<a href="#{s["id"]}">{s["titulo"]}</a>' for s in sections
    )
    sidebar_html = f"""
    <div id="sidebar">
      <h2>🐾 PetFinder EDA</h2>
      {sidebar_links}
    </div>
    """

    # Construcción del contenido principal con grids de gráficos por sección
    sections_html = ''
    for sec in sections:
        cards_html = ''
        for fig, fig_title in sec['figs']:
            if fig is None:
                continue
            # Convertir figura Plotly a HTML div (sin la librería JS, se incluye una sola vez)
            fig_html = pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs=False,
                config={'responsive': True, 'displayModeBar': True}
            )
            cards_html += f"""
            <div class="chart-card">
              <div class="chart-card-title">{fig_title}</div>
              {fig_html}
            </div>
            """

        sections_html += f"""
        <div class="section" id="{sec['id']}">
          <div class="section-title">{sec['titulo']}</div>
          <div class="charts-grid">
            {cards_html}
          </div>
        </div>
        """

    # Ensamblado del HTML final completo
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PetFinder EDA — Dashboard Interactivo</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  {css}
</head>
<body>
  {sidebar_html}
  <div id="main">
    <h1>🐾 PetFinder EDA — Dashboard Interactivo</h1>
    <p class="subtitle">
      Dataset: PetFinder.my Adoption Prediction (Kaggle) &nbsp;|&nbsp;
      ~14,000 mascotas &nbsp;|&nbsp; Malasia &nbsp;|&nbsp;
      Objetivo: predecir velocidad de adopción
    </p>
    {sections_html}
  </div>
  {js}
</body>
</html>
"""
    return html

# Generar y guardar el dashboard
dashboard_html = build_dashboard_html(dashboard_sections)
dashboard_path = f'{OUTPUT_PATH}/dashboard_eda_petfinder.html'

with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(dashboard_html)

print(f"\n✅ Dashboard interactivo guardado en: {dashboard_path}")
print("   Abrí el archivo en cualquier navegador para explorar todos los gráficos.")
