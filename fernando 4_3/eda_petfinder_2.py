import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import missingno as msno
from wordcloud import WordCloud
from scipy.stats import pearsonr, kruskal, chi2_contingency
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title='PetFinder — EDA Interactivo', layout='wide')

st.markdown("""<style>
body, .stMarkdown, p { font-family: Georgia, serif; }
h1, h2, h3, h4 { font-family: Georgia, serif; color: #2c3e50; }
.stTabs [data-baseweb="tab"] { font-family: Georgia, serif; font-size: 1rem; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }
.stDataFrame { font-family: Georgia, serif; }
</style>""", unsafe_allow_html=True)

BASE = Path('/Users/fernandopaganini/Desktop/Laboratorio 2/UA_MDM_Labo2/input/petfinder-adoption-prediction')
OUTPUT = Path('/Users/fernandopaganini/Desktop/Laboratorio 2/output')
OUTPUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.18, 'grid.color': '#cccccc',
    'axes.edgecolor': '#aaaaaa', 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.labelcolor': '#333333',
    'xtick.color': '#555555', 'ytick.color': '#555555',
    'legend.frameon': False,
})

CBLUE  = '#4a6fa5'
CRED   = '#c0392b'
CGRAY  = '#7f8c8d'
CGOLD  = '#d4a843'
CGREEN = '#27ae60'

ADOPTION_LABELS = {0: 'Mismo día', 1: '1-7 días', 2: '8-30 días', 3: '31-90 días', 4: '>100 días'}
ADOPTION_ORDER  = [ADOPTION_LABELS[i] for i in range(5)]


def save_fig(fig, name):
    fig.savefig(OUTPUT / f'{name}.png', dpi=150, bbox_inches='tight', facecolor='white')


def top_n(series, n=9):
    counts = series.value_counts()
    if len(counts) <= n:
        return counts
    return pd.concat([counts.iloc[:n], pd.Series({'Otros': counts.iloc[n:].sum()})])


def make_bar(data, title, figsize=(7, 4), color=None, horizontal=False):
    color = color or CBLUE
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        ax.barh(range(len(data)), data.values, color=color, alpha=0.82)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([str(x) for x in data.index], fontsize=9)
        ax.set_xlabel('Cantidad')
        ax.invert_yaxis()
    else:
        ax.bar(range(len(data)), data.values, color=color, alpha=0.82)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([str(x) for x in data.index], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Cantidad')
    ax.set_title(title, fontsize=12, pad=8)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    plt.tight_layout()
    return fig


def make_boxplot(series, title, ylabel='', figsize=(5, 5), color=None):
    color = color or CBLUE
    clean = series.dropna()
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(clean, patch_artist=True,
               medianprops={'color': CRED, 'linewidth': 2},
               boxprops={'facecolor': color, 'alpha': 0.6},
               whiskerprops={'color': CGRAY}, capprops={'color': CGRAY},
               flierprops={'marker': 'o', 'markerfacecolor': CGRAY,
                           'markersize': 3, 'alpha': 0.35, 'markeredgewidth': 0})
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    plt.tight_layout()
    return fig


def make_assoc_bar(x_labels, y_vals, title, xlabel, ylabel, figsize=(8, 4), color=None):
    color = color or CBLUE
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(x_labels)), y_vals, color=color, alpha=0.82)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([str(x) for x in x_labels], rotation=45, ha='right', fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    plt.tight_layout()
    return fig


@st.cache_data
def load_main():
    breeds = pd.read_csv(BASE / 'breed_labels.csv')
    colors = pd.read_csv(BASE / 'color_labels.csv')
    states = pd.read_csv(BASE / 'state_labels.csv')
    bmap = {**dict(zip(breeds['BreedID'], breeds['BreedName'])), 0: 'Desconocido'}
    cmap = {**dict(zip(colors['ColorID'], colors['ColorName'])), 0: 'Ninguno'}
    smap = dict(zip(states['StateID'], states['StateName']))
    df = pd.read_csv(BASE / 'train' / 'train.csv')
    df['Type']        = df['Type'].map({1: 'Perro', 2: 'Gato'})
    df['Gender']      = df['Gender'].map({1: 'Macho', 2: 'Hembra', 3: 'Mixto'})
    df['Breed1']      = df['Breed1'].map(bmap).fillna('Desconocido')
    df['Breed2']      = df['Breed2'].map(bmap).fillna('Desconocido')
    df['State']       = df['State'].map(smap)
    df['MaturitySize']= df['MaturitySize'].map({1: 'Pequeño', 2: 'Mediano', 3: 'Grande', 4: 'X-Grande'})
    df['FurLength']   = df['FurLength'].map({1: 'Corto', 2: 'Mediano', 3: 'Largo'})
    df['Vaccinated']  = df['Vaccinated'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    df['Dewormed']    = df['Dewormed'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    df['Sterilized']  = df['Sterilized'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    df['Health']      = df['Health'].map({1: 'Sano', 2: 'Lesión leve', 3: 'Lesión grave'})
    df['Negro']       = df['Color1'].map(cmap)
    df['Marrón']      = df['Color2'].map(cmap)
    df['Dorado']      = df['Color3'].map(cmap)
    df = df.drop(columns=['Name', 'Description', 'Color1', 'Color2', 'Color3', 'RescuerID'])
    return df


@st.cache_data
def load_metadata():
    meta_dir = BASE / 'train_metadata'
    records, all_labels = [], []
    for fp in sorted(meta_dir.glob('*.json')):
        pet_id = fp.stem.rsplit('-', 1)[0]
        try:
            data = json.loads(fp.read_text(encoding='utf-8'))
        except Exception:
            continue
        lbls = data.get('labelAnnotations', [])
        for lb in lbls:
            all_labels.append({'PetID': pet_id,
                                'description': lb.get('description', ''),
                                'score': lb.get('score', 0),
                                'topicality': lb.get('topicality', 0)})
        dr = dg = db = mpf = np.nan
        try:
            clrs = data['imagePropertiesAnnotation']['dominantColors']['colors']
            pfs  = [c.get('pixelFraction', 0) for c in clrs]
            tot  = sum(pfs)
            if tot > 0:
                dr  = sum(c['color'].get('red',   0) * p for c, p in zip(clrs, pfs)) / tot
                dg  = sum(c['color'].get('green', 0) * p for c, p in zip(clrs, pfs)) / tot
                db  = sum(c['color'].get('blue',  0) * p for c, p in zip(clrs, pfs)) / tot
                mpf = max(pfs)
        except (KeyError, TypeError):
            pass
        cc = np.nan
        try:
            hints = data['cropHintsAnnotation']['cropHints']
            if hints:
                cc = float(np.mean([h.get('confidence', 0) for h in hints]))
        except (KeyError, TypeError):
            pass
        records.append({
            'PetID': pet_id,
            'n_labels': len(lbls),
            'avg_label_score': float(np.mean([lb.get('score', 0) for lb in lbls])) if lbls else np.nan,
            'dom_R': dr, 'dom_G': dg, 'dom_B': db,
            'max_pixelFraction': mpf,
            'crop_confidence': cc,
            'has_face': 1 if data.get('faceAnnotations') else 0,
            'has_text': 1 if data.get('textAnnotations') else 0,
        })
    if not records:
        return pd.DataFrame(), pd.DataFrame()
    mdf = pd.DataFrame(records)
    agg = mdf.groupby('PetID').agg(
        n_labels_img     =('n_labels',          'mean'),
        avg_label_score  =('avg_label_score',   'mean'),
        dom_R            =('dom_R',             'mean'),
        dom_G            =('dom_G',             'mean'),
        dom_B            =('dom_B',             'mean'),
        max_pixelFraction=('max_pixelFraction', 'mean'),
        crop_confidence  =('crop_confidence',   'mean'),
        has_face         =('has_face',          'max'),
        has_text         =('has_text',          'max'),
    ).reset_index()
    return agg, pd.DataFrame(all_labels)


@st.cache_data
def load_sentiment():
    sent_dir = BASE / 'train_sentiment'
    records, all_entities = [], []
    for fp in sent_dir.glob('*.json'):
        pet_id = fp.stem
        try:
            data = json.loads(fp.read_text(encoding='utf-8'))
        except Exception:
            continue
        ds    = data.get('documentSentiment', {})
        score = ds.get('score', np.nan)
        mag   = ds.get('magnitude', np.nan)
        sents = data.get('sentences', [])
        s_range = np.nan
        if len(sents) > 1:
            ss = [s.get('sentiment', {}).get('score', 0) for s in sents]
            s_range = max(ss) - min(ss)
        for ent in data.get('entities', []):
            all_entities.append({'PetID': pet_id,
                                  'name':    ent.get('name', ''),
                                  'type':    ent.get('type', 'OTHER'),
                                  'salience':ent.get('salience', 0)})
        try:
            sc = 'Positiva' if score > 0.25 else ('Negativa' if score < -0.25 else 'Neutra')
        except TypeError:
            sc = 'Neutra'
        records.append({'PetID': pet_id, 'doc_score': score, 'doc_magnitude': mag,
                         'n_sentences': len(sents), 'sentence_score_range': s_range,
                         'sentiment_class': sc})
    return pd.DataFrame(records), pd.DataFrame(all_entities)


@st.cache_data
def compute_associations(_df):
    num_cols = _df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in _df.select_dtypes(include='object').columns if c != 'PetID']
    pairs = []

    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i + 1:]:
            valid = _df[[c1, c2]].dropna()
            if len(valid) < 10:
                continue
            r = valid[c1].corr(valid[c2])
            if not np.isnan(r):
                pairs.append({'var1': c1, 'var2': c2, 'tipo': 'num-num',
                               'medida': abs(float(r)), 'medida_raw': float(r)})

    for cat in cat_cols:
        if _df[cat].nunique() > 30:
            continue
        for num in num_cols:
            valid = _df[[cat, num]].dropna()
            if len(valid) < 10 or valid[cat].nunique() < 2:
                continue
            groups     = [valid[num][valid[cat] == c].values for c in valid[cat].unique()]
            grand_mean = valid[num].mean()
            ss_b = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups if len(g) > 0)
            ss_t = ((valid[num] - grand_mean) ** 2).sum()
            eta  = float(np.sqrt(ss_b / ss_t)) if ss_t > 0 else np.nan
            if not np.isnan(eta):
                pairs.append({'var1': cat, 'var2': num, 'tipo': 'cat-num',
                               'medida': eta, 'medida_raw': eta})

    for i, c1 in enumerate(cat_cols):
        if _df[c1].nunique() > 30:
            continue
        for c2 in cat_cols[i + 1:]:
            if _df[c2].nunique() > 30:
                continue
            valid = _df[[c1, c2]].dropna()
            if len(valid) < 10:
                continue
            try:
                ct        = pd.crosstab(valid[c1], valid[c2])
                chi2_val, _, _, _ = chi2_contingency(ct)
                n         = int(ct.values.sum())
                min_dim   = min(ct.shape) - 1
                if min_dim > 0 and n > 0:
                    cv = float(np.sqrt(chi2_val / (n * min_dim)))
                    if not np.isnan(cv):
                        pairs.append({'var1': c1, 'var2': c2, 'tipo': 'cat-cat',
                                       'medida': cv, 'medida_raw': cv})
            except Exception:
                pass

    return pd.DataFrame(pairs).sort_values('medida', ascending=False).reset_index(drop=True)


with st.spinner('Cargando datos… (la primera ejecución puede tardar varios minutos al procesar ~58 000 archivos de metadata)'):
    df_main             = load_main()
    meta_agg, labels_df = load_metadata()
    sent_df, ent_df     = load_sentiment()

df = df_main.copy()
if not meta_agg.empty:
    df = df.merge(meta_agg, on='PetID', how='left')
if not sent_df.empty:
    df = df.merge(sent_df,  on='PetID', how='left')

HAS_META = 'avg_label_score' in df.columns
HAS_SENT = 'doc_score'       in df.columns

st.title('Análisis Exploratorio de Datos — PetFinder Adoption Prediction')

tab1, tab2, tab3 = st.tabs(['📊  Distribución', '🔗  Asociación', '✅  Significación'])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DISTRIBUCIÓN
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header('Estructura del Dataset')

    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric('Registros totales',        f'{len(df):,}')
    cm2.metric('Variables',                str(df.shape[1] - 1))
    cm3.metric('Con metadata de imágenes', f"{df['has_face'].notna().sum():,}" if HAS_META else 'N/A')
    cm4.metric('Con análisis de sentimiento', f"{df['doc_score'].notna().sum():,}" if HAS_SENT else 'N/A')

    st.markdown('---')
    st.subheader('Tipos de datos por variable')
    df_disp = df.drop(columns=['PetID'], errors='ignore')
    dtype_tbl = pd.DataFrame({
        'Variable':         df_disp.columns,
        'Tipo':             df_disp.dtypes.astype(str).values,
        'Registros no nulos': df_disp.notna().sum().values,
        '% completo':       (df_disp.notna().mean() * 100).round(1).values,
    })
    st.dataframe(dtype_tbl, hide_index=True, use_container_width=True)

    st.markdown('---')
    st.subheader('Valores nulos por variable')
    null_cnt = df_disp.isnull().sum()
    null_pct = (null_cnt / len(df) * 100).round(2)
    null_tbl = pd.DataFrame({
        'Variable': df_disp.columns,
        'Valores nulos': null_cnt.values,
        'Porcentaje (%)': null_pct.values,
    }).query('`Valores nulos` > 0').sort_values('Valores nulos', ascending=False).reset_index(drop=True)
    if not null_tbl.empty:
        st.dataframe(null_tbl, hide_index=True, use_container_width=True)
    else:
        st.success('No se encontraron valores nulos en el dataset.')

    st.markdown('**Visualización con missingno**')
    ca, cb = st.columns(2)
    with ca:
        st.write('Matriz de nulos')
        plt.figure(figsize=(11, 6))
        msno.matrix(df_disp, sparkline=False, color=(0.29, 0.44, 0.65))
        fig_mm = plt.gcf()
        plt.tight_layout()
        st.pyplot(fig_mm)
        save_fig(fig_mm, 'msno_matrix')
        plt.close()
    with cb:
        st.write('Completitud por variable')
        plt.figure(figsize=(11, 6))
        msno.bar(df_disp, color=CBLUE, fontsize=8)
        fig_mb = plt.gcf()
        plt.tight_layout()
        st.pyplot(fig_mb)
        save_fig(fig_mb, 'msno_bar')
        plt.close()

    st.markdown('---')
    st.subheader('Estadísticas descriptivas — Variables numéricas')
    num_stat_cols = df_disp.select_dtypes(include=[np.number]).columns.tolist()
    stat_rows = []
    for col in num_stat_cols:
        s = df_disp[col].dropna()
        if len(s) == 0:
            continue
        moda = s.mode()
        stat_rows.append({
            'Variable':   col,
            'Media':      round(float(s.mean()), 4),
            'Mediana':    round(float(s.median()), 4),
            'Mínimo':     round(float(s.min()), 4),
            'Máximo':     round(float(s.max()), 4),
            'Moda':       round(float(moda.iloc[0]), 4) if not moda.empty else np.nan,
            'Desv. Est.': round(float(s.std()), 4),
        })
    st.dataframe(pd.DataFrame(stat_rows), hide_index=True, use_container_width=True)

    st.markdown('---')
    st.markdown('## Distribuciones')

    st.markdown('### Variable Objetivo — Velocidad de Adopción')
    adop_counts = df['AdoptionSpeed'].map(ADOPTION_LABELS).value_counts()
    adop_ord    = adop_counts.reindex(ADOPTION_ORDER, fill_value=0)
    cl, _ = st.columns([2, 1])
    with cl:
        fig = make_bar(adop_ord, 'Distribución de AdoptionSpeed', figsize=(9, 4), color=CRED)
        st.pyplot(fig)
        save_fig(fig, 'dist_adoption_speed')
        plt.close()

    st.markdown('---')
    st.markdown('### Variables generales')

    c1, c2 = st.columns(2)
    with c1:
        fig = make_bar(top_n(df['Type']),   'Tipo de animal', figsize=(5, 4))
        st.pyplot(fig); save_fig(fig, 'dist_type'); plt.close()
    with c2:
        fig = make_bar(top_n(df['Gender']), 'Género',         figsize=(5, 4))
        st.pyplot(fig); save_fig(fig, 'dist_gender'); plt.close()

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = make_boxplot(df['Age'],      'Edad (meses)',         'Meses')
        st.pyplot(fig); save_fig(fig, 'box_age'); plt.close()
    with c2:
        fig = make_boxplot(df['Fee'],      'Tarifa de adopción',   'Monto')
        st.pyplot(fig); save_fig(fig, 'box_fee'); plt.close()
    with c3:
        fig = make_boxplot(df['Quantity'], 'Cantidad de animales', 'Cantidad')
        st.pyplot(fig); save_fig(fig, 'box_quantity'); plt.close()

    c1, c2 = st.columns(2)
    with c1:
        fig = make_boxplot(df['PhotoAmt'], 'Número de fotos',  'Fotos')
        st.pyplot(fig); save_fig(fig, 'box_photoamt'); plt.close()
    with c2:
        fig = make_boxplot(df['VideoAmt'], 'Número de videos', 'Videos')
        st.pyplot(fig); save_fig(fig, 'box_videoamt'); plt.close()

    c1, c2 = st.columns(2)
    with c1:
        fig = make_bar(top_n(df['MaturitySize']), 'Tamaño de madurez',   figsize=(5, 4))
        st.pyplot(fig); save_fig(fig, 'dist_maturitysize'); plt.close()
    with c2:
        fig = make_bar(top_n(df['FurLength']),    'Longitud del pelaje', figsize=(5, 4))
        st.pyplot(fig); save_fig(fig, 'dist_furlength'); plt.close()

    cl, _ = st.columns([1, 1])
    with cl:
        fig = make_bar(top_n(df['Health']), 'Estado de salud', figsize=(5, 4))
        st.pyplot(fig); save_fig(fig, 'dist_health'); plt.close()

    fig = make_bar(top_n(df['State']), 'Estado (Malasia)', figsize=(9, 4), horizontal=True)
    st.pyplot(fig); save_fig(fig, 'dist_state'); plt.close()

    c1, c2 = st.columns(2)
    with c1:
        fig = make_bar(top_n(df['Breed1']), 'Raza principal (Top 9)',   figsize=(7, 5), horizontal=True)
        st.pyplot(fig); save_fig(fig, 'dist_breed1'); plt.close()
    with c2:
        fig = make_bar(top_n(df['Breed2']), 'Raza secundaria (Top 9)',  figsize=(7, 5), horizontal=True)
        st.pyplot(fig); save_fig(fig, 'dist_breed2'); plt.close()

    st.markdown('---')
    st.markdown('### Grupo: Color')
    c1, c2, c3 = st.columns(3)
    for col_name, cw in [('Negro', c1), ('Marrón', c2), ('Dorado', c3)]:
        with cw:
            fig = make_bar(top_n(df[col_name]), f'Color — {col_name}', figsize=(5, 4))
            st.pyplot(fig)
            save_fig(fig, f'dist_color_{col_name.lower().replace("ó","o").replace("á","a")}')
            plt.close()

    st.markdown('---')
    st.markdown('### Grupo: Salud')
    c1, c2, c3 = st.columns(3)
    for col_name, label_es, cw in [('Vaccinated', 'Vacunado', c1),
                                    ('Dewormed',   'Desparasitado', c2),
                                    ('Sterilized', 'Esterilizado', c3)]:
        with cw:
            fig = make_bar(top_n(df[col_name]), label_es, figsize=(5, 4))
            st.pyplot(fig); save_fig(fig, f'dist_{col_name.lower()}'); plt.close()

    if HAS_META:
        st.markdown('---')
        st.markdown('### Grupo: Metadata de imágenes')

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = make_boxplot(df['n_labels_img'],      'Etiquetas por imagen (prom.)', 'N° etiquetas')
            st.pyplot(fig); save_fig(fig, 'box_n_labels_img'); plt.close()
        with c2:
            fig = make_boxplot(df['avg_label_score'],   'Score promedio de etiquetas',  'Score')
            st.pyplot(fig); save_fig(fig, 'box_avg_label_score'); plt.close()
        with c3:
            fig = make_boxplot(df['max_pixelFraction'], 'Fracción píxeles dominante',   'Fracción')
            st.pyplot(fig); save_fig(fig, 'box_max_pixel_fraction'); plt.close()

        c1, c2 = st.columns(2)
        with c1:
            fig = make_boxplot(df['crop_confidence'], 'Confianza de recorte (cropHint)', 'Confianza')
            st.pyplot(fig); save_fig(fig, 'box_crop_confidence'); plt.close()
        with c2:
            crop_hi  = (df['crop_confidence'] > 0.8).mean() * 100
            face_pct = df['has_face'].mean() * 100
            text_pct = df['has_text'].mean() * 100
            ann_data = pd.Series({
                'CropHint confianza > 0.8': round(crop_hi,  1),
                'Con anotación facial':     round(face_pct, 1),
                'Con anotación de texto':   round(text_pct, 1),
            })
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.barh(range(len(ann_data)), ann_data.values,
                           color=[CBLUE, CRED, CGRAY], alpha=0.82)
            ax.set_yticks(range(len(ann_data)))
            ax.set_yticklabels(ann_data.index, fontsize=9)
            for bar, val in zip(bars, ann_data.values):
                ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', va='center', fontsize=9)
            ax.set_xlabel('% de mascotas')
            ax.set_title('Presencia de anotaciones en imágenes', fontsize=11)
            ax.spines['left'].set_alpha(0.3); ax.spines['bottom'].set_alpha(0.3)
            plt.tight_layout()
            st.pyplot(fig); save_fig(fig, 'pct_annotations'); plt.close()

        valid_rgb = df[['dom_R', 'dom_G', 'dom_B']].dropna()
        if len(valid_rgb) > 0:
            ar = valid_rgb['dom_R'].mean() / 255
            ag = valid_rgb['dom_G'].mean() / 255
            ab = valid_rgb['dom_B'].mean() / 255
            cl, _ = st.columns([1, 3])
            with cl:
                fig, ax = plt.subplots(figsize=(3, 2))
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=(ar, ag, ab)))
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
                ax.set_title(
                    f'Color promedio del dataset\nRGB({int(ar*255)}, {int(ag*255)}, {int(ab*255)})',
                    fontsize=9)
                plt.tight_layout()
                st.pyplot(fig); save_fig(fig, 'avg_dominant_color'); plt.close()

        if not labels_df.empty:
            st.markdown('#### Top 20 etiquetas más frecuentes')
            lbl_freq = labels_df['description'].value_counts().head(20)
            fig = make_bar(lbl_freq, 'Top 20 etiquetas de imágenes',
                           figsize=(10, 5), horizontal=True)
            st.pyplot(fig); save_fig(fig, 'top20_labels'); plt.close()

            st.markdown('#### Score vs Topicality por etiqueta (Top 15 por frecuencia)')
            top15 = labels_df['description'].value_counts().head(15).index.tolist()
            lbl_st = (labels_df[labels_df['description'].isin(top15)]
                      .groupby('description')
                      .agg(score_mean=('score', 'mean'), topicality_mean=('topicality', 'mean'))
                      .reindex(top15))
            fig, ax = plt.subplots(figsize=(11, 5))
            xp  = np.arange(len(lbl_st))
            w   = 0.35
            ax.bar(xp - w / 2, lbl_st['score_mean'],     w, label='Score',     color=CBLUE, alpha=0.82)
            ax.bar(xp + w / 2, lbl_st['topicality_mean'],w, label='Topicality', color=CGOLD, alpha=0.82)
            ax.set_xticks(xp)
            ax.set_xticklabels(lbl_st.index, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Valor promedio'); ax.legend(fontsize=9)
            ax.set_title('Score vs Topicality promedio por etiqueta (Top 15)', fontsize=12)
            ax.spines['left'].set_alpha(0.3); ax.spines['bottom'].set_alpha(0.3)
            plt.tight_layout()
            st.pyplot(fig); save_fig(fig, 'score_vs_topicality'); plt.close()

            st.markdown('#### Distribución de pixelFraction del color dominante por imagen')
            fig = make_boxplot(df['max_pixelFraction'],
                               'pixelFraction del color dominante (por mascota, promedio)',
                               'Fracción', figsize=(7, 4))
            cl, _ = st.columns([2, 1])
            with cl:
                st.pyplot(fig); save_fig(fig, 'box_pixelFraction_dist'); plt.close()

            st.markdown('#### Nube de palabras — Etiquetas ponderadas por score promedio')
            lbl_wc = labels_df.groupby('description')['score'].mean().to_dict()
            if lbl_wc:
                wc = WordCloud(background_color='white', width=900, height=420,
                               colormap='Blues', prefer_horizontal=0.85, min_font_size=8)
                wc.generate_from_frequencies(lbl_wc)
                fig, ax = plt.subplots(figsize=(11, 5))
                ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                ax.set_title('Etiquetas de imágenes — score promedio', fontsize=12, pad=10)
                plt.tight_layout()
                st.pyplot(fig); save_fig(fig, 'wc_labels'); plt.close()

    if HAS_SENT:
        st.markdown('---')
        st.markdown('### Grupo: Sentimientos')

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = make_boxplot(df['doc_score'],    'Score global de sentimiento', 'Score (−1 a +1)')
            st.pyplot(fig); save_fig(fig, 'box_doc_score'); plt.close()
        with c2:
            fig = make_boxplot(df['doc_magnitude'],'Magnitud global de sentimiento','Magnitud')
            st.pyplot(fig); save_fig(fig, 'box_doc_magnitude'); plt.close()
        with c3:
            fig = make_boxplot(df['n_sentences'],  'Número de oraciones',        'Oraciones')
            st.pyplot(fig); save_fig(fig, 'box_n_sentences'); plt.close()

        c1, c2 = st.columns(2)
        with c1:
            fig = make_boxplot(df['sentence_score_range'], 'Variación emocional interna', 'Rango de score')
            st.pyplot(fig); save_fig(fig, 'box_sent_range'); plt.close()
        with c2:
            fig = make_bar(top_n(df['sentiment_class']), 'Clasificación de sentimiento', figsize=(5, 4))
            st.pyplot(fig); save_fig(fig, 'dist_sentiment_class'); plt.close()

        st.markdown('#### Score de sentimiento por tipo de animal')
        vt = df[['Type', 'doc_score']].dropna()
        if not vt.empty:
            types_s = sorted(vt['Type'].unique())
            fig, ax = plt.subplots(figsize=(6, 4))
            data_bp = [vt[vt['Type'] == t]['doc_score'].values for t in types_s]
            bp = ax.boxplot(data_bp, patch_artist=True, labels=types_s,
                            medianprops={'color': CRED, 'linewidth': 2},
                            boxprops={'alpha': 0.7},
                            whiskerprops={'color': CGRAY}, capprops={'color': CGRAY})
            for patch, col in zip(bp['boxes'], [CBLUE, CGOLD]):
                patch.set_facecolor(col)
            ax.set_ylabel('Score de sentimiento')
            ax.set_title('Score de sentimiento por tipo de animal', fontsize=12)
            ax.spines['left'].set_alpha(0.3); ax.spines['bottom'].set_alpha(0.3)
            plt.tight_layout()
            st.pyplot(fig); save_fig(fig, 'sentiment_by_type'); plt.close()

        if not ent_df.empty:
            st.markdown('#### Entidades más frecuentes por tipo')
            c1, c2, c3 = st.columns(3)
            for et, cw, label_es in [('PERSON', c1, 'Personas'),
                                      ('LOCATION', c2, 'Lugares'),
                                      ('OTHER', c3, 'Otros')]:
                sub = ent_df[ent_df['type'] == et]
                if len(sub) > 0:
                    top_e = sub.groupby('name')['salience'].mean().sort_values(ascending=False).head(10)
                    with cw:
                        fig = make_bar(top_e, f'Entidades: {label_es}',
                                       figsize=(5, 5), horizontal=True)
                        st.pyplot(fig); save_fig(fig, f'ent_{et.lower()}'); plt.close()

            st.markdown('#### Nube de palabras — Entidades ponderadas por salience promedio')
            ent_wc = ent_df.groupby('name')['salience'].mean().to_dict()
            if ent_wc:
                wc = WordCloud(background_color='white', width=900, height=420,
                               colormap='Greys', prefer_horizontal=0.8)
                wc.generate_from_frequencies(ent_wc)
                fig, ax = plt.subplots(figsize=(11, 5))
                ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                ax.set_title('Entidades — salience promedio', fontsize=12, pad=10)
                plt.tight_layout()
                st.pyplot(fig); save_fig(fig, 'wc_entities'); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ASOCIACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header('Análisis de Asociación')

    num_all = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_m  = df[num_all].corr()

    st.subheader('Tabla de correlación — Variables numéricas')
    st.dataframe(corr_m.round(3), use_container_width=True)

    st.subheader('Heatmap de correlación')
    nv   = len(corr_m)
    fsz  = max(10, nv * 0.7)
    fig, ax = plt.subplots(figsize=(fsz, fsz * 0.85))
    im   = ax.imshow(corr_m.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(nv))
    ax.set_yticks(range(nv))
    ax.set_xticklabels(corr_m.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr_m.columns, fontsize=8)
    for i in range(nv):
        for j in range(nv):
            val = corr_m.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6.5, color='white' if abs(val) > 0.5 else '#333333')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_title('Matriz de correlación — variables numéricas', fontsize=13, pad=12)
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); save_fig(fig, 'heatmap_correlacion'); plt.close()

    st.markdown('---')
    st.markdown('## Asociación de variables con AdoptionSpeed')

    def _cat_vs_adoption(col_name, top=None):
        valid = df[[col_name, 'AdoptionSpeed']].dropna()
        if top:
            tc = valid[col_name].value_counts().head(top).index
            valid = valid[valid[col_name].isin(tc)]
        ma = valid.groupby(col_name)['AdoptionSpeed'].mean().sort_values()
        return ma.index.tolist(), ma.values

    def _num_vs_adoption(col_name):
        valid = df[[col_name, 'AdoptionSpeed']].dropna()
        mb = valid.groupby('AdoptionSpeed')[col_name].mean()
        lbs = [ADOPTION_LABELS.get(i, str(i)) for i in mb.index]
        return lbs, mb.values

    st.markdown('### Variables generales')

    for col_name, ylabel_str in [('Age', 'Edad prom. (meses)'),
                                   ('Fee', 'Tarifa prom.'),
                                   ('Quantity', 'Cantidad prom.'),
                                   ('PhotoAmt', 'Fotos prom.'),
                                   ('VideoAmt', 'Videos prom.')]:
        xl, yv = _num_vs_adoption(col_name)
        fig = make_assoc_bar(xl, yv, f'{col_name} promedio por velocidad de adopción',
                              'Velocidad de adopción', ylabel_str)
        cl, _ = st.columns([2, 1])
        with cl:
            st.pyplot(fig); save_fig(fig, f'assoc_{col_name.lower()}_adoption'); plt.close()

    for col_name, lbl in [('Type', 'Tipo de animal'), ('Gender', 'Género'),
                           ('MaturitySize', 'Tamaño de madurez'), ('FurLength', 'Longitud pelaje'),
                           ('Health', 'Estado de salud')]:
        xl, yv = _cat_vs_adoption(col_name)
        fig = make_assoc_bar(xl, yv, f'AdoptionSpeed promedio — {lbl}', lbl,
                              'AdoptionSpeed (promedio)')
        cl, _ = st.columns([2, 1])
        with cl:
            st.pyplot(fig); save_fig(fig, f'assoc_{col_name.lower()}_adoption'); plt.close()

    for col_name, lbl, topk in [('State', 'Estado', 10),
                                  ('Breed1', 'Raza principal', 9),
                                  ('Breed2', 'Raza secundaria', 9)]:
        xl, yv = _cat_vs_adoption(col_name, top=topk)
        fig = make_assoc_bar(xl, yv, f'AdoptionSpeed promedio — {lbl} (Top {topk})',
                              lbl, 'AdoptionSpeed (promedio)', figsize=(9, 4))
        st.pyplot(fig); save_fig(fig, f'assoc_{col_name.lower()}_adoption'); plt.close()

    st.markdown('---')
    st.markdown('### Grupo: Color')
    for col_name in ['Negro', 'Marrón', 'Dorado']:
        xl, yv = _cat_vs_adoption(col_name)
        fig = make_assoc_bar(xl, yv, f'AdoptionSpeed promedio — Color {col_name}',
                              col_name, 'AdoptionSpeed (promedio)', figsize=(7, 4))
        cl, _ = st.columns([2, 1])
        with cl:
            st.pyplot(fig)
            save_fig(fig, f'assoc_color_{col_name.lower().replace("ó","o").replace("á","a")}_adoption')
            plt.close()

    st.markdown('---')
    st.markdown('### Grupo: Salud')
    for col_name, lbl in [('Vaccinated', 'Vacunado'), ('Dewormed', 'Desparasitado'),
                           ('Sterilized', 'Esterilizado')]:
        xl, yv = _cat_vs_adoption(col_name)
        fig = make_assoc_bar(xl, yv, f'AdoptionSpeed promedio — {lbl}',
                              lbl, 'AdoptionSpeed (promedio)', figsize=(7, 4))
        cl, _ = st.columns([2, 1])
        with cl:
            st.pyplot(fig); save_fig(fig, f'assoc_{col_name.lower()}_adoption'); plt.close()

    if HAS_META:
        st.markdown('---')
        st.markdown('### Grupo: Metadata de imágenes')
        for col_name, lbl in [('n_labels_img', 'Etiquetas por imagen (prom.)'),
                               ('avg_label_score', 'Score prom. de etiquetas'),
                               ('max_pixelFraction', 'Fracción píxeles dominante'),
                               ('crop_confidence', 'Confianza de recorte')]:
            if col_name not in df.columns:
                continue
            xl, yv = _num_vs_adoption(col_name)
            fig = make_assoc_bar(xl, yv, f'{lbl} por velocidad de adopción',
                                  'Velocidad de adopción', f'{lbl} (promedio)')
            cl, _ = st.columns([2, 1])
            with cl:
                st.pyplot(fig); save_fig(fig, f'assoc_{col_name}_adoption'); plt.close()

        for col_name, lbl in [('has_face', 'Tiene anotación facial'),
                               ('has_text', 'Tiene anotación de texto')]:
            if col_name not in df.columns:
                continue
            valid = df[[col_name, 'AdoptionSpeed']].dropna()
            valid[col_name] = valid[col_name].map({0: 'No', 1: 'Sí'})
            ma = valid.groupby(col_name)['AdoptionSpeed'].mean().sort_values()
            fig = make_assoc_bar(ma.index.tolist(), ma.values,
                                  f'AdoptionSpeed promedio — {lbl}',
                                  lbl, 'AdoptionSpeed (promedio)', figsize=(5, 4))
            cl, _ = st.columns([2, 1])
            with cl:
                st.pyplot(fig); save_fig(fig, f'assoc_{col_name}_adoption'); plt.close()

    if HAS_SENT:
        st.markdown('---')
        st.markdown('### Grupo: Sentimientos')
        for col_name, lbl in [('doc_score', 'Score global'),
                               ('doc_magnitude', 'Magnitud global'),
                               ('n_sentences', 'N° de oraciones'),
                               ('sentence_score_range', 'Variación emocional interna')]:
            if col_name not in df.columns:
                continue
            xl, yv = _num_vs_adoption(col_name)
            fig = make_assoc_bar(xl, yv, f'{lbl} por velocidad de adopción',
                                  'Velocidad de adopción', f'{lbl} (promedio)')
            cl, _ = st.columns([2, 1])
            with cl:
                st.pyplot(fig); save_fig(fig, f'assoc_{col_name}_adoption'); plt.close()

        if 'sentiment_class' in df.columns:
            xl, yv = _cat_vs_adoption('sentiment_class')
            fig = make_assoc_bar(xl, yv,
                                  'AdoptionSpeed promedio por clasificación de sentimiento',
                                  'Clase de sentimiento', 'AdoptionSpeed (promedio)', figsize=(7, 4))
            cl, _ = st.columns([2, 1])
            with cl:
                st.pyplot(fig); save_fig(fig, 'assoc_sentiment_class_adoption'); plt.close()

            st.markdown('#### Cruce: Clasificación de sentimiento × AdoptionSpeed')
            valid_sc = df[['sentiment_class', 'AdoptionSpeed']].dropna()
            cross    = pd.crosstab(valid_sc['sentiment_class'],
                                   valid_sc['AdoptionSpeed'].map(ADOPTION_LABELS),
                                   normalize='index') * 100
            cross    = cross.reindex(columns=ADOPTION_ORDER, fill_value=0)
            fig, ax  = plt.subplots(figsize=(9, 4))
            xp       = np.arange(len(ADOPTION_ORDER))
            w        = 0.25
            clrs_sc  = [CBLUE, CGOLD, CRED]
            for idx, (cls_name, row) in enumerate(cross.iterrows()):
                ax.bar(xp + idx * w, [row.get(lbl, 0) for lbl in ADOPTION_ORDER],
                       w, label=cls_name, color=clrs_sc[idx % 3], alpha=0.82)
            ax.set_xticks(xp + w)
            ax.set_xticklabels(ADOPTION_ORDER, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('% dentro de cada clase de sentimiento')
            ax.set_title('Distribución de AdoptionSpeed por clase de sentimiento', fontsize=12)
            ax.legend(fontsize=9)
            ax.spines['left'].set_alpha(0.3); ax.spines['bottom'].set_alpha(0.3)
            plt.tight_layout()
            st.pyplot(fig); save_fig(fig, 'cross_sentiment_adoption'); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — SIGNIFICACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header('Análisis de Significación Estadística')

    with st.spinner('Calculando asociaciones entre todas las variables…'):
        pairs_df = compute_associations(df)

    THRESHOLD = 0.30
    sig_pairs = pairs_df[pairs_df['medida'] > THRESHOLD].copy()

    if sig_pairs.empty:
        st.warning(f'No se encontraron asociaciones con |coeficiente| > {THRESHOLD}. '
                   f'Se muestran las 10 más fuertes.')
        work_pairs = pairs_df.head(10)
    else:
        st.success(f'Se encontraron **{len(sig_pairs)}** pares con |coeficiente| > {THRESHOLD}.')
        work_pairs = sig_pairs

    results = []
    for _, row in work_pairs.iterrows():
        v1, v2, tipo = row['var1'], row['var2'], row['tipo']
        medida       = row['medida_raw']
        test_name    = stat_v = p_v = np.nan

        if tipo == 'num-num':
            test_name = 'Pearson'
            valid = df[[v1, v2]].dropna()
            if len(valid) >= 10:
                try:
                    stat_v, p_v = pearsonr(valid[v1], valid[v2])
                except Exception:
                    pass

        elif tipo == 'cat-num':
            test_name = 'Kruskal-Wallis'
            valid = df[[v1, v2]].dropna()
            if len(valid) >= 10 and valid[v1].nunique() >= 2:
                groups = [valid[v2][valid[v1] == c].values
                          for c in valid[v1].unique()
                          if len(valid[v2][valid[v1] == c]) > 0]
                if len(groups) >= 2:
                    try:
                        stat_v, p_v = kruskal(*groups)
                    except Exception:
                        pass

        elif tipo == 'cat-cat':
            test_name = 'Chi-cuadrado'
            valid = df[[v1, v2]].dropna()
            if len(valid) >= 10:
                try:
                    ct = pd.crosstab(valid[v1], valid[v2])
                    stat_v, p_v, _, _ = chi2_contingency(ct)
                except Exception:
                    pass

        try:
            interp = ('Significativo (α=0.05)' if float(p_v) < 0.05
                      else 'No significativo')
        except (TypeError, ValueError):
            interp = 'N/A'

        def fmt(x, dec=4):
            try:
                v = float(x)
                return round(v, dec) if not np.isnan(v) else 'N/A'
            except (TypeError, ValueError):
                return 'N/A'

        def fmt_p(x):
            try:
                v = float(x)
                return f'{v:.4e}' if not np.isnan(v) else 'N/A'
            except (TypeError, ValueError):
                return 'N/A'

        tipo_legible = {'num-num': 'Numérica — Numérica',
                        'cat-num': 'Categórica — Numérica',
                        'cat-cat': 'Categórica — Categórica'}.get(tipo, tipo)

        results.append({
            'Variable 1':       v1,
            'Variable 2':       v2,
            'Tipo de par':      tipo_legible,
            'Test aplicado':    test_name,
            'Coeficiente':      fmt(medida),
            'Estadístico':      fmt(stat_v),
            'p-valor':          fmt_p(p_v),
            'Interpretación':   interp,
        })

    res_df = pd.DataFrame(results)
    st.dataframe(res_df, hide_index=True, use_container_width=True)

    n_sig   = (res_df['Interpretación'] == 'Significativo (α=0.05)').sum()
    n_total = len(res_df)
    st.markdown(f'**{n_sig} de {n_total} pares evaluados son estadísticamente significativos (α = 0.05)**')

    if not sig_pairs.empty:
        st.markdown('---')
        st.markdown('#### Distribución de coeficientes de asociación (pares con |coef| > 0.30)')
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(sig_pairs['medida'].values, bins=20, color=CBLUE, alpha=0.82, edgecolor='white')
        ax.axvline(THRESHOLD, color=CRED, linewidth=1.5, linestyle='--')
        ax.set_xlabel('|Coeficiente de asociación|')
        ax.set_ylabel('N° de pares')
        ax.set_title('Distribución de asociaciones significativas', fontsize=12)
        ax.spines['left'].set_alpha(0.3); ax.spines['bottom'].set_alpha(0.3)
        plt.tight_layout()
        st.pyplot(fig); save_fig(fig, 'hist_asociaciones_significativas'); plt.close()
