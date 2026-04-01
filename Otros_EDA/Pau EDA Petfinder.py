# Si alguien quiere entrar a mi drive para ver el reporte interactivo file:///C:/Users/paula/AppData/Local/Programs/Microsoft%20VS%20Code/eda_output/petfinder_profile_report.html

from pathlib import Path
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# =========================
# 0) SETTINGS (EDIT HERE)
# =========================
DATA_DIR = Path(r"C:\Users\paula\OneDrive\Maestria\Lab II\Carpeta VCS\Petfinder competicion\Competicion\Train")

# If your file is not train.csv, change it here:
PREFERRED_FILE = "train.csv"

# These are the usual Petfinder competition column names:
TARGET = "AdoptionSpeed"   # what we want to analyze/predict
PET_KIND = "Type"          # kind of pet (often 1=Dog, 2=Cat)

OUT_DIR = Path("eda_output")
OUT_DIR.mkdir(exist_ok=True)

# =========================
# 1) LOAD DATA
# =========================
preferred_path = DATA_DIR / PREFERRED_FILE

if preferred_path.exists():
    data_path = preferred_path
else:
    # beginner-friendly fallback: choose the first CSV in the folder
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found in:\n{DATA_DIR}")
    data_path = csv_files[0]

df = pd.read_csv(data_path)
print(f"Loaded file: {data_path}")
print("Dataset shape (rows, columns):", df.shape)

# Safety checks so you immediately see what to fix if column names differ
if TARGET not in df.columns:
    raise ValueError(f"TARGET column '{TARGET}' not found. Columns are:\n{list(df.columns)}")

if PET_KIND not in df.columns:
    raise ValueError(f"PET_KIND column '{PET_KIND}' not found. Columns are:\n{list(df.columns)}")

# =========================
# 2) QUICK OVERVIEW TABLES
# =========================
overview = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_%": (df.isna().mean() * 100).round(2),
    "n_unique": df.nunique(dropna=True)
}).sort_values(["missing_%", "n_unique"], ascending=[False, False])

overview.to_csv(OUT_DIR / "overview_columns.csv", index=True)

# =========================
# 3) PLOTS (SAVED TO PNG)
# =========================
def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()

# (A) Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x=TARGET)
plt.title(f"Distribution of {TARGET}")
savefig(OUT_DIR / "fig_01_target_distribution.png")

# (B) Missing values plot (top 30 columns)
missing = (df.isna().mean() * 100).sort_values(ascending=False)
missing_top = missing[missing > 0].head(30)

if len(missing_top) > 0:
    plt.figure(figsize=(10, 5))
    missing_top.plot(kind="bar")
    plt.title("Top 30 columns with missing values (%)")
    plt.ylabel("Missing %")
    savefig(OUT_DIR / "fig_02_missing_top30.png")

# (C) AdoptionSpeed by pet kind (boxplot)
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x=PET_KIND, y=TARGET)
plt.title(f"{TARGET} by {PET_KIND}")
savefig(OUT_DIR / "fig_03_target_by_pet_kind.png")

# (D) AdoptionSpeed fractions within each pet kind (stacked bars)
ct = pd.crosstab(df[PET_KIND], df[TARGET], normalize="index")
ax = ct.plot(kind="bar", stacked=True, figsize=(10, 5))
ax.set_title(f"{TARGET} distribution within each {PET_KIND} (fractions)")
ax.set_ylabel("Fraction")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_04_pet_kind_stacked.png", dpi=170)
plt.close()

# =========================
# 4) FEATURE RELATIONSHIP WITH ADOPTION SPEED
#    (Simple beginner approach: numeric correlations)
# =========================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != TARGET]

# Correlation with target (screening)
corr_with_target = df[numeric_cols + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
corr_with_target = corr_with_target.reindex(corr_with_target.abs().sort_values(ascending=False).index)

corr_with_target.to_csv(OUT_DIR / "numeric_corr_with_target.csv")

# Plot top 6 numeric features vs target (boxplots by AdoptionSpeed)
top6 = corr_with_target.head(6).index.tolist()

for i, col in enumerate(top6, start=1):
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x=TARGET, y=col)
    plt.title(f"{col} vs {TARGET}")
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", col)
    savefig(OUT_DIR / f"fig_10_top_numeric_{i}_{safe}.png")

# =========================
# 5) COMPREHENSIVE AUTOMATIC REPORT (HTML)
# =========================
# This creates a large, detailed EDA report automatically.
from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Petfinder Comprehensive EDA", explorative=True)
profile_html = OUT_DIR / "petfinder_profile_report.html"
profile.to_file(profile_html)

# =========================
# 6) YOUR OWN SIMPLE SUMMARY REPORT (HTML -> PDF)
# =========================
summary_html = OUT_DIR / "petfinder_eda_summary.html"
summary_pdf = OUT_DIR / "petfinder_eda_summary.pdf"

target_counts = df[TARGET].value_counts().sort_index()

top_missing_table = overview.head(20)
pet_kind_counts = df[PET_KIND].value_counts(dropna=False)

# Write a beginner-friendly HTML report (easy to print / convert to PDF)
with open(summary_html, "w", encoding="utf-8") as f:
    f.write(f"""
<html>
<head>
  <meta charset="utf-8"/>
  <title>Petfinder EDA Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; }}
    h1, h2 {{ color: #1f2d3d; }}
    .box {{ background: #f3f6f9; padding: 12px; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; }}
    th {{ background: #fafafa; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #eee; }}
    code {{ background: #f3f3f3; padding: 2px 4px; }}
  </style>
</head>
<body>

<h1>Petfinder EDA Summary (Beginner-Friendly)</h1>

<div class="box">
  <p><b>Goal:</b> Understand how features relate to <code>{TARGET}</code>, comparing different pet kinds (<code>{PET_KIND}</code>).</p>
  <p><b>Data loaded from:</b> {data_path}</p>
  <p><b>Rows:</b> {df.shape[0]} &nbsp;&nbsp; <b>Columns:</b> {df.shape[1]}</p>
</div>

<h2>1) What is the target?</h2>
<p><code>{TARGET}</code> is the adoption speed class (in the Petfinder competition it is usually an ordinal class).</p>
{target_counts.to_frame("count").to_html()}

<img src="fig_01_target_distribution.png" alt="Target distribution" />

<h2>2) Pet kinds (dogs/cats/etc.)</h2>
<p>This shows how many records you have per pet kind (<code>{PET_KIND}</code>).</p>
{pet_kind_counts.to_frame("count").to_html()}

<img src="fig_03_target_by_pet_kind.png" alt="Target by pet kind" />
<img src="fig_04_pet_kind_stacked.png" alt="Stacked fractions by pet kind" />

<h2>3) Data quality: missing values</h2>
<p>Missing values can affect models and conclusions. Here are the columns with the most missing values.</p>
{top_missing_table.to_html()}

{"<img src='fig_02_missing_top30.png' alt='Missing values top 30' />" if len(missing_top) > 0 else "<p>No missing values found.</p>"}

<h2>4) Which numeric features relate most to adoption speed?</h2>
<p>We compute a simple correlation between each numeric feature and <code>{TARGET}</code>. This is a first screening (not final causation).</p>
<p><b>Top correlated numeric features:</b></p>
{corr_with_target.head(12).to_frame("corr_with_target").to_html()}

<p><b>Plots for the top numeric features:</b></p>
{"".join([f"<img src='fig_10_top_numeric_{i}_{re.sub(r'[^a-zA-Z0-9_-]+','_',c)}.png' />" for i, c in enumerate(top6, start=1)])}

<h2>5) Comprehensive report (automatic)</h2>
<p>A full automatic EDA report was created as: <code>petfinder_profile_report.html</code>.</p>

</body>
</html>
""")

print(f"✓ Summary HTML saved to: {summary_html}")
