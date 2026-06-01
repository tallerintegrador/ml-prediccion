"""
03b_clasificacion.py — Detección de anomalías y clasificación de riesgo en facturas.

Metodología: Isolation Forest (sin supervisión) para etiquetar anomalías,
             XGBoost multiclase para clasificación rápida en producción.
Prerequisito: python python/02_feature_engineering.py
Ejecutar    : python python/03b_clasificacion.py
Salidas     : models/isolation_forest.joblib
              models/xgboost_clasificador.joblib
              models/xgboost_label_encoder.joblib
              models/riesgo_encoder.joblib
              models/riesgo_score_params.joblib
              reports/figures/clasificacion_feature_importance.png
"""

# --- celda 1 ---
# Imports
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# --- celda 2 ---
# Rutas y carga de datos
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
FIGURES_DIR   = BASE_DIR / "reports" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_parquet(PROCESSED_DIR / "dataset_modelable_train.parquet")
test  = pd.read_parquet(PROCESSED_DIR / "dataset_modelable_test.parquet")
feature_config = joblib.load(MODELS_DIR / "feature_config.joblib")

FEATURES_NUM = [f for f in feature_config["features_num"] if f in train.columns]
FEATURES_CAT = [f for f in feature_config["features_cat"] if f in train.columns]
ALL_FEATURES = FEATURES_NUM + FEATURES_CAT

# Excluir el target y columnas auxiliares (no son features de modelo)
FEATURES_MODELO = [
    f for f in ALL_FEATURES
    if f not in ["fecha_imputada", "sample_weight", "log_target", "es_outlier"]
]

print(f"Train: {len(train):,} filas | Test: {len(test):,} filas")
print(f"Features para clasificación: {len(FEATURES_MODELO)}")

# --- celda 3 ---
# Encoding ordinal para Isolation Forest (no acepta strings)
print("\n=== ENCODING ORDINAL PARA ISOLATION FOREST ===")

cat_features_if = [f for f in FEATURES_CAT if f in FEATURES_MODELO]
num_features_if = [f for f in FEATURES_NUM if f in FEATURES_MODELO]

riesgo_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
    encoded_missing_value=-1,
)
riesgo_encoder.fit(train[cat_features_if].astype(str))

def preparar_X_if(df):
    """Combina features numéricas + categóricas codificadas ordinalmente."""
    X_num = df[num_features_if].fillna(0).values
    X_cat = riesgo_encoder.transform(df[cat_features_if].astype(str))
    return np.hstack([X_num, X_cat])

X_train_if = preparar_X_if(train)
X_test_if  = preparar_X_if(test)
print(f"  Matriz de features IF: train={X_train_if.shape}, test={X_test_if.shape}")

# --- celda 4 ---
# Isolation Forest: detección de anomalías sin supervisión
print("\n=== ISOLATION FOREST ===")

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.10,   # esperamos ~10% de facturas anómalas
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)
iso_forest.fit(X_train_if)

# Scores en train: decision_function (más negativo = más anómalo)
train_scores = iso_forest.decision_function(X_train_if)
test_scores  = iso_forest.decision_function(X_test_if)

# Invertir para que mayor valor = mayor anomalía
train["anomaly_score_raw"] = -train_scores
test["anomaly_score_raw"]  = -test_scores

print(f"  Anomalías detectadas en train: {(iso_forest.predict(X_train_if) == -1).sum():,} "
      f"({(iso_forest.predict(X_train_if) == -1).mean()*100:.1f}%)")
print(f"  Anomalías detectadas en test : {(iso_forest.predict(X_test_if) == -1).sum():,} "
      f"({(iso_forest.predict(X_test_if) == -1).mean()*100:.1f}%)")

# --- celda 5 ---
# Asignación de niveles de riesgo y normalización del score a 0-100
print("\n=== ETIQUETADO DE RIESGO ===")

# Umbrales fijos calculados sobre el train y aplicados al test.
# Calcular percentiles dentro de cada conjunto por separado producía una
# distribución artificialmente uniforme en test (siempre 50/30/20%).
def asignar_nivel_riesgo(scores_array, umbral_p50=None, umbral_p80=None):
    """
    Asigna ALTO/MEDIO/BAJO usando umbrales absolutos del score.
    Si umbral_p50/p80 son None, los calcula sobre la propia serie (modo train).
    Devuelve también los umbrales para reutilizarlos en test.
    """
    scores = np.asarray(scores_array, dtype=float)
    if umbral_p50 is None or umbral_p80 is None:
        umbral_p50 = float(np.percentile(scores, 50))
        umbral_p80 = float(np.percentile(scores, 80))
    niveles   = np.where(scores > umbral_p80, "ALTO",
                np.where(scores > umbral_p50, "MEDIO", "BAJO"))
    percentil = pd.Series(scores).rank(pct=True).values * 100
    return percentil, niveles, umbral_p50, umbral_p80

train_percentil, train_nivel, umbral_p50_train, umbral_p80_train = asignar_nivel_riesgo(
    train["anomaly_score_raw"].values
)
test_percentil, test_nivel, _, _ = asignar_nivel_riesgo(
    test["anomaly_score_raw"].values, umbral_p50_train, umbral_p80_train
)

train["riesgo_percentil"] = train_percentil
train["nivel_riesgo"]     = train_nivel
test["riesgo_percentil"]  = test_percentil
test["nivel_riesgo"]      = test_nivel

riesgo_score_params = {
    "min":         float(train["anomaly_score_raw"].min()),
    "max":         float(train["anomaly_score_raw"].max()),
    "umbral_p50":  umbral_p50_train,
    "umbral_p80":  umbral_p80_train,
}

# Normalizar score a 0-100 usando los parámetros del train
score_min = riesgo_score_params["min"]
score_max = riesgo_score_params["max"]

train["riesgo_score_0_100"] = (
    (train["anomaly_score_raw"] - score_min) /
    (score_max - score_min + 1e-9) * 100
).clip(0, 100)
test["riesgo_score_0_100"] = (
    (test["anomaly_score_raw"] - score_min) /
    (score_max - score_min + 1e-9) * 100
).clip(0, 100)

print("Distribución de niveles en train:")
print(pd.Series(train_nivel.astype(str)).value_counts().to_string())
print("\nDistribución de niveles en test:")
print(pd.Series(test_nivel.astype(str)).value_counts().to_string())

print(f"\nRiesgo score train — mediana: {train['riesgo_score_0_100'].median():.1f} | "
      f"P90: {train['riesgo_score_0_100'].quantile(0.9):.1f}")

# --- celda 6 ---
# XGBoost multiclase: aprende a predecir los niveles asignados por IF
# Más rápido que IF en producción y portátil vía API
print("\n=== XGBOOST MULTICLASE ===")

le = LabelEncoder()
le.fit(["BAJO", "MEDIO", "ALTO"])

y_train_enc = le.transform(train["nivel_riesgo"])
y_test_enc  = le.transform(test["nivel_riesgo"])

# Features para XGBoost: las mismas que IF pero en formato dataframe (acepta strings via DMatrix)
def preparar_X_xgb(df):
    """Features numéricas + categóricas como strings para XGBoost."""
    X = df[FEATURES_MODELO].copy()
    for col in [c for c in FEATURES_MODELO if c in cat_features_if]:
        X[col] = X[col].astype(str)
    for col in [c for c in FEATURES_MODELO if c in num_features_if]:
        X[col] = X[col].fillna(0)
    return X

X_train_xgb = preparar_X_xgb(train)
X_test_xgb  = preparar_X_xgb(test)

# Codificar categóricas como ordinales para XGBoost también
cat_encoders_xgb = {}
for col in cat_features_if:
    if col in X_train_xgb.columns:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_xgb[col] = enc.fit_transform(X_train_xgb[[col]])
        X_test_xgb[col]  = enc.transform(X_test_xgb[[col]])
        cat_encoders_xgb[col] = enc

# TimeSeriesSplit CV
tscv = TimeSeriesSplit(n_splits=5, gap=15)
acc_folds = []

for idx_tr, idx_val in tscv.split(X_train_xgb):
    X_f, X_v = X_train_xgb.iloc[idx_tr], X_train_xgb.iloc[idx_val]
    y_f, y_v = y_train_enc[idx_tr], y_train_enc[idx_val]
    w_f = compute_sample_weight("balanced", y_f)

    m = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        verbosity=0,
        random_state=42,
    )
    m.fit(X_f, y_f, sample_weight=w_f, eval_set=[(X_v, y_v)], verbose=False)
    acc_folds.append(accuracy_score(y_v, m.predict(X_v)))

print(f"  Accuracy CV (5 folds): {np.mean(acc_folds):.3f} ± {np.std(acc_folds):.3f}")

# Modelo final con todos los datos de train
w_train = compute_sample_weight("balanced", y_train_enc)
xgb_final = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    use_label_encoder=False,
    verbosity=0,
    random_state=42,
)
xgb_final.fit(X_train_xgb, y_train_enc, sample_weight=w_train, verbose=False)

# --- celda 7 ---
# Evaluación en test
print("\n=== EVALUACIÓN EN TEST ===")

y_pred_test = xgb_final.predict(X_test_xgb)
y_pred_labels = le.inverse_transform(y_pred_test)
y_real_labels = le.inverse_transform(y_test_enc)

acc = accuracy_score(y_test_enc, y_pred_test)
print(f"  Accuracy en test: {acc:.3f}")

y_pred_proba = xgb_final.predict_proba(X_test_xgb)
roc_auc = roc_auc_score(y_test_enc, y_pred_proba, multi_class="ovr", average="macro")
print(f"  ROC-AUC macro (OVR): {roc_auc:.3f}")

print(f"\n  Reporte de clasificación:")
print(classification_report(y_real_labels, y_pred_labels))

# Error más costoso: facturas ALTO clasificadas como BAJO
cm_labels = ["BAJO", "MEDIO", "ALTO"]
cm_full   = confusion_matrix(y_real_labels, y_pred_labels, labels=cm_labels)
alto_idx  = cm_labels.index("ALTO")
bajo_idx  = cm_labels.index("BAJO")
err_criticos = int(cm_full[alto_idx, bajo_idx])
total_alto   = int(cm_full[alto_idx, :].sum())
print(f"  Errores críticos (ALTO→BAJO): {err_criticos}/{total_alto} "
      f"({err_criticos / max(total_alto, 1) * 100:.1f}%)")

# Matriz de confusión
cm = confusion_matrix(y_real_labels, y_pred_labels, labels=["BAJO", "MEDIO", "ALTO"])
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BAJO", "MEDIO", "ALTO"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Clasificación de riesgo — Matriz de confusión (test)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clasificacion_confusion_matrix.png", dpi=120)
plt.close()

# Feature importance (top 15)
importancia = pd.Series(
    xgb_final.feature_importances_,
    index=X_train_xgb.columns
).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 6))
importancia.plot.barh(ax=ax, color="teal", edgecolor="white")
ax.invert_yaxis()
ax.set_xlabel("Importancia (F-score)")
ax.set_title("Top 15 features — Clasificador de riesgo XGBoost")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clasificacion_feature_importance.png", dpi=120)
plt.close()

print(f"\nTop 10 features por importancia:")
print(importancia.head(10).to_string())
print("Figuras guardadas en reports/figures/")

# --- celda 8 ---
# Guardar todos los artefactos
joblib.dump(iso_forest,           MODELS_DIR / "isolation_forest.joblib")
joblib.dump(xgb_final,            MODELS_DIR / "xgboost_clasificador.joblib")
joblib.dump(le,                   MODELS_DIR / "xgboost_label_encoder.joblib")
joblib.dump(riesgo_encoder,       MODELS_DIR / "riesgo_encoder.joblib")
joblib.dump(riesgo_score_params,  MODELS_DIR / "riesgo_score_params.joblib")
joblib.dump(cat_encoders_xgb,     MODELS_DIR / "riesgo_cat_encoders_xgb.joblib")

print(f"\nModelos guardados en {MODELS_DIR}:")
for f in sorted(MODELS_DIR.glob("*.joblib")):
    print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
