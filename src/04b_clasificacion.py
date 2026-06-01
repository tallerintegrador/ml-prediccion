"""
04b_clasificacion.py — CRISP-DM Fase 4 (Track B): Clasificación de riesgo por desvío.

Metodología CORRECTA (sin data leakage):
  1. Target: nivel de riesgo BAJO/MEDIO/ALTO basado en desvío porcentual
             del costo real respecto a la tarifa histórica (EWM con shift=1)
  2. Garantía anti-leakage: 'tarifa_historica' y sus derivados se excluyen
             de las features del clasificador porque se usaron para construir el target
  3. Comparación objetiva: RandomForest, GradientBoosting, XGBoost → ganador por F1-macro CV
  4. Evaluación: accuracy, F1 por clase, matriz de confusión, error crítico (ALTO→BAJO)

Prerequisito: python src/03_feature_engineering.py
Ejecutar    : python src/04b_clasificacion.py
Salidas     : models/clasificador_riesgo.joblib
              models/label_encoder_riesgo.joblib
              models/riesgo_score_params.joblib
              models/riesgo_cat_encoders.joblib
              reports/figures/clasificacion_confusion_matrix.png
              reports/figures/clasificacion_feature_importance.png
              reports/figures/clasificacion_comparacion_algoritmos.png
"""

import sys
import warnings
from pathlib import Path

import copy
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, RANDOM_STATE, DPI_FIGURAS

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
train = pd.read_parquet(PROCESSED_DIR / "dataset_modelable_train.parquet")
test  = pd.read_parquet(PROCESSED_DIR / "dataset_modelable_test.parquet")
feature_config = joblib.load(MODELS_DIR / "feature_config.joblib")

FEATURES_NUM = [f for f in feature_config["features_num"] if f in train.columns]
FEATURES_CAT = [f for f in feature_config["features_cat"] if f in train.columns]
ALL_FEATURES = FEATURES_NUM + FEATURES_CAT

print(f"Train: {len(train):,} filas | Test: {len(test):,} filas")

# ---------------------------------------------------------------------------
# PASO 1: Construcción del target por desvío porcentual
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("PASO 1: CONSTRUCCIÓN DEL TARGET POR DESVÍO (sin leakage)")
print("=" * 65)

# tarifa_historica fue calculada con shift(1) en FE → no contamina la observación actual
# Regla anti-leakage: tarifa_historica usada para construir el target NO puede
# usarse como predictor (ni sus derivados: diff_tarifa_mediana, tarifa_por_contenedor)

UMBRAL_TARIFA_MINIMA = 1.0  # PEN — evita divisiones por cero o tarifas despreciables

for df_part in [train, test]:
    tarifa_ref_pen = np.expm1(df_part["tarifa_historica"])
    real_pen       = df_part["importe_total_pen"]

    # Desvío porcentual absoluto respecto a la tarifa histórica
    df_part["_desvio_pct"] = np.abs(
        (real_pen - tarifa_ref_pen) / tarifa_ref_pen.clip(lower=UMBRAL_TARIFA_MINIMA)
    ) * 100

# Umbrales calibrados SOLO en train (P50 y P80 del desvío absoluto)
umbral_bajo  = float(train["_desvio_pct"].quantile(0.50))
umbral_medio = float(train["_desvio_pct"].quantile(0.80))

print(f"  Umbral BAJO  (P50): {umbral_bajo:.1f}%")
print(f"  Umbral MEDIO (P80): {umbral_medio:.1f}%")
print(f"  Criterio: BAJO ≤ {umbral_bajo:.1f}% | MEDIO ≤ {umbral_medio:.1f}% | ALTO > {umbral_medio:.1f}%")

def asignar_nivel(desvio_pct, ub, um):
    return np.where(desvio_pct > um, "ALTO",
           np.where(desvio_pct > ub, "MEDIO", "BAJO"))

train["nivel_riesgo_desvio"] = asignar_nivel(train["_desvio_pct"], umbral_bajo, umbral_medio)
test["nivel_riesgo_desvio"]  = asignar_nivel(test["_desvio_pct"],  umbral_bajo, umbral_medio)

print("\n  Distribución del target en train:")
for nivel, n in train["nivel_riesgo_desvio"].value_counts().items():
    pct = n / len(train) * 100
    print(f"    {nivel:<6}: {n:>6,} ({pct:.1f}%)")

print("\n  Distribución del target en test:")
for nivel, n in test["nivel_riesgo_desvio"].value_counts().items():
    pct = n / len(test) * 100
    print(f"    {nivel:<6}: {n:>6,} ({pct:.1f}%)")

riesgo_score_params = {
    "umbral_bajo":  umbral_bajo,
    "umbral_medio": umbral_medio,
    "criterio":     "desvio_porcentual_absoluto_vs_tarifa_historica",
}

# ---------------------------------------------------------------------------
# PASO 2: Definición de features (excluir las usadas para construir el target)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("PASO 2: FEATURES DEL CLASIFICADOR (anti-leakage)")
print("=" * 65)

# Excluir features que fueron usadas para derivar el target
# tarifa_historica → construyó desvio_pct → EXCLUIDA
# diff_tarifa_mediana = tarifa_historica - mediana → EXCLUIDA (contiene tarifa_historica)
# tarifa_por_contenedor → EWM similar a tarifa_historica → EXCLUIDA
FEATURES_EXCLUIR = {
    "tarifa_historica", "diff_tarifa_mediana", "tarifa_por_contenedor",
    "sample_weight", "log_target", "es_outlier", "fecha_imputada", "_desvio_pct",
}

FEATURES_NUM_CLS = [f for f in FEATURES_NUM if f not in FEATURES_EXCLUIR]
FEATURES_CAT_CLS = [f for f in FEATURES_CAT if f not in FEATURES_EXCLUIR]
FEATURES_CLS     = FEATURES_NUM_CLS + FEATURES_CAT_CLS

print(f"  Features numéricas: {len(FEATURES_NUM_CLS)} (excluidas: tarifa_historica, diff_tarifa_mediana, tarifa_por_contenedor)")
print(f"  Features categóricas: {len(FEATURES_CAT_CLS)}")
print(f"  Total features clasificador: {len(FEATURES_CLS)}")

# Codificación ordinal (todos los algoritmos la necesitan)
cat_cols_cls = FEATURES_CAT_CLS
num_cols_cls = FEATURES_NUM_CLS

cat_encoders = {}
def preparar_X(df_sub, fit=False):
    """Prepara la matriz de features codificando categóricas ordinalmente."""
    X = df_sub[FEATURES_CLS].copy()
    for col in cat_cols_cls:
        X[col] = X[col].astype(str).fillna("DESCONOCIDO")
    for col in num_cols_cls:
        X[col] = X[col].fillna(0)
    if fit:
        for col in cat_cols_cls:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[col] = enc.fit_transform(X[[col]]).ravel()
            cat_encoders[col] = enc
    else:
        for col in cat_cols_cls:
            enc = cat_encoders[col]
            X[col] = enc.transform(X[[col]]).ravel()
    return X.values.astype(float)

# Label encoder: BAJO=0, MEDIO=1, ALTO=2
le = LabelEncoder()
le.fit(["BAJO", "MEDIO", "ALTO"])

y_train = le.transform(train["nivel_riesgo_desvio"])
y_test  = le.transform(test["nivel_riesgo_desvio"])

X_train_cls = preparar_X(train, fit=True)
X_test_cls  = preparar_X(test,  fit=False)

print(f"\n  Matriz train: {X_train_cls.shape} | test: {X_test_cls.shape}")

# ---------------------------------------------------------------------------
# PASO 3: Comparación de algoritmos (3-fold TimeSeriesSplit)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("PASO 3: COMPARACIÓN DE ALGORITMOS (3-Fold TimeSeriesSplit, F1-macro)")
print("=" * 65)

tscv_cls = TimeSeriesSplit(n_splits=3)
candidatos = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, random_state=RANDOM_STATE,
    ),
    "XGBoost": xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=250, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", verbosity=0, random_state=RANDOM_STATE,
    ),
}

resultados_cls = {nombre: [] for nombre in candidatos}

for fold_idx, (idx_tr, idx_val) in enumerate(tscv_cls.split(X_train_cls)):
    X_f, X_v = X_train_cls[idx_tr], X_train_cls[idx_val]
    y_f, y_v = y_train[idx_tr], y_train[idx_val]
    w_f      = compute_sample_weight("balanced", y_f)

    for nombre, clf in candidatos.items():
        clf_fold = copy.deepcopy(clf)
        clf_fold.fit(X_f, y_f, sample_weight=w_f)
        y_pred   = clf_fold.predict(X_v)
        f1_macro = f1_score(y_v, y_pred, average="macro")
        resultados_cls[nombre].append(f1_macro)
        print(f"  Fold {fold_idx+1} | {nombre:<20}: F1-macro = {f1_macro:.3f}")

resumen_f1 = {n: np.mean(v) for n, v in resultados_cls.items()}
ganador_cls = max(resumen_f1, key=resumen_f1.get)

print(f"\nF1-macro promedio (3-fold):")
for nombre, f1_avg in sorted(resumen_f1.items(), key=lambda x: -x[1]):
    marca = " ← GANADOR" if nombre == ganador_cls else ""
    print(f"  {nombre:<22}: {f1_avg:.3f}{marca}")

# Gráfico de comparación
fig, ax = plt.subplots(figsize=(8, 5))
nombres = list(resumen_f1.keys())
medias  = [resumen_f1[n] for n in nombres]
stds    = [np.std(resultados_cls[n]) for n in nombres]
colores = ["mediumseagreen" if n == ganador_cls else "steelblue" for n in nombres]
bars    = ax.bar(nombres, medias, yerr=stds, capsize=6, color=colores, edgecolor="white", linewidth=1.2)
ax.set_ylim(0, 1)
ax.set_ylabel("F1-macro promedio", fontsize=11)
ax.set_title(f"Comparación de clasificadores — F1-macro (3-Fold TS-CV)", fontsize=13, fontweight="bold")
for bar, m in zip(bars, medias):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{m:.3f}", ha="center", fontsize=10)
ax.axhline(0.70, color="orange", linestyle="--", linewidth=1, label="Referencia 0.70")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clasificacion_comparacion_algoritmos.png", dpi=DPI_FIGURAS)
plt.close()
print("Guardado: clasificacion_comparacion_algoritmos.png")

# ---------------------------------------------------------------------------
# PASO 4: Entrenamiento final con el algoritmo ganador
# ---------------------------------------------------------------------------
print(f"\n" + "=" * 65)
print(f"PASO 4: MODELO FINAL — {ganador_cls}")
print("=" * 65)

w_train_all = compute_sample_weight("balanced", y_train)
clf_final   = candidatos[ganador_cls]

if ganador_cls == "XGBoost":
    clf_final = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=350, max_depth=5, learning_rate=0.07,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", verbosity=0, random_state=RANDOM_STATE,
    )
    clf_final.fit(X_train_cls, y_train, sample_weight=w_train_all, verbose=False)
elif ganador_cls == "RandomForest":
    clf_final = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf_final.fit(X_train_cls, y_train)
else:  # GradientBoosting
    clf_final = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.07,
        subsample=0.8, random_state=RANDOM_STATE,
    )
    clf_final.fit(X_train_cls, y_train, sample_weight=w_train_all)

# ---------------------------------------------------------------------------
# PASO 5: Evaluación completa en test
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("PASO 5: EVALUACIÓN EN TEST")
print("=" * 65)

y_pred_test   = clf_final.predict(X_test_cls)
y_pred_labels = le.inverse_transform(y_pred_test)
y_real_labels = le.inverse_transform(y_test)

acc     = accuracy_score(y_test, y_pred_test)
f1_mac  = f1_score(y_test, y_pred_test, average="macro")
f1_cada = f1_score(y_test, y_pred_test, average=None, labels=[0, 1, 2])

print(f"  Accuracy en test       : {acc:.3f}")
print(f"  F1-macro en test       : {f1_mac:.3f}")
print(f"  F1 BAJO                : {f1_cada[0]:.3f}")
print(f"  F1 MEDIO               : {f1_cada[1]:.3f}")
print(f"  F1 ALTO                : {f1_cada[2]:.3f}")
print(f"\n  Reporte completo:")
print(classification_report(y_real_labels, y_pred_labels, target_names=["BAJO", "MEDIO", "ALTO"]))

# Error crítico: facturas ALTO clasificadas como BAJO
cm_full   = confusion_matrix(y_real_labels, y_pred_labels, labels=["BAJO", "MEDIO", "ALTO"])
err_crit  = int(cm_full[2, 0])   # ALTO real → BAJO predicho (fila ALTO, col BAJO)
total_alto = int(cm_full[2, :].sum())
print(f"  Error crítico (ALTO→BAJO): {err_crit}/{total_alto} ({err_crit/max(total_alto,1)*100:.1f}%)")
print(f"  Interpretación: facturas de alta desviación incorrectamente marcadas como normales")

# Limitaciones
print("\n  LIMITACIONES DEL MODELO:")
print("  1. El target depende de la calidad de las tarifas históricas (primeros proveedores tienen mayor incertidumbre)")
print("  2. Concept drift: proveedores con cambios de precio abruptos pueden eludir la clasificación")
print("  3. Umbrales calibrados sobre datos 2020-2024; revisar periódicamente con nuevos datos")

# Gráfico: Matriz de confusión
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_full, display_labels=["BAJO", "MEDIO", "ALTO"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Matriz de confusión — {ganador_cls} (test)\n"
             f"Accuracy={acc:.2f}, F1-macro={f1_mac:.2f}", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "clasificacion_confusion_matrix.png", dpi=DPI_FIGURAS)
plt.close()
print("\nGuardado: clasificacion_confusion_matrix.png")

# Gráfico: Importancia de features
if hasattr(clf_final, "feature_importances_"):
    importancia = pd.Series(clf_final.feature_importances_, index=FEATURES_CLS)
    importancia = importancia.sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    colores_imp = ["coral" if f in {"tarifa_historica", "diff_tarifa_mediana"} else "steelblue"
                   for f in importancia.index]
    importancia.plot.barh(ax=ax, color=colores_imp, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Importancia (F-score)", fontsize=11)
    ax.set_title(f"Top 15 features — Clasificador de riesgo ({ganador_cls})", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "clasificacion_feature_importance.png", dpi=DPI_FIGURAS)
    plt.close()
    print("Guardado: clasificacion_feature_importance.png")

    print(f"\nTop 10 features más importantes:")
    for feat, imp in importancia.head(10).items():
        print(f"  {feat:<35}: {imp:.4f}")

# ---------------------------------------------------------------------------
# Guardar artefactos
# ---------------------------------------------------------------------------
joblib.dump(clf_final,           MODELS_DIR / "clasificador_riesgo.joblib")
joblib.dump(le,                  MODELS_DIR / "label_encoder_riesgo.joblib")
joblib.dump(riesgo_score_params, MODELS_DIR / "riesgo_score_params.joblib")
joblib.dump(cat_encoders,        MODELS_DIR / "riesgo_cat_encoders.joblib")
joblib.dump({
    "features_cls":     FEATURES_CLS,
    "features_num_cls": FEATURES_NUM_CLS,
    "features_cat_cls": FEATURES_CAT_CLS,
    "algoritmo":        ganador_cls,
    "f1_macro_cv":      resumen_f1[ganador_cls],
    "accuracy_test":    acc,
    "f1_macro_test":    f1_mac,
}, MODELS_DIR / "clasificacion_config.joblib")

print(f"\nModelos guardados en {MODELS_DIR}:")
for f in sorted(MODELS_DIR.glob("clasificador*.joblib")) + sorted(MODELS_DIR.glob("label_encoder*.joblib")) + sorted(MODELS_DIR.glob("riesgo*.joblib")) + sorted(MODELS_DIR.glob("clasificacion*.joblib")):
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
print("\nClasificación finalizada. Próximo paso: python src/04c_clustering.py")
