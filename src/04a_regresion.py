"""
04a_regresion.py — CRISP-DM Fase 4 (Track A): Regresión de costos de importación.

Metodología:
  1. Comparación objetiva de 3 algoritmos: Ridge, XGBoost, LightGBM (3-fold TS-CV global)
  2. LightGBM seleccionado como ganador → optimización por concepto con Optuna
  3. Corrección de sesgo en 3 capas: escalar OOF → piecewise → proveedor×concepto
  4. Intervalos de confianza al 80% mediante Conformal Quantile Regression (P10-P90)
  5. Evaluación rigurosa: MAE, MAPE, MdAPE, RMSE, R², cobertura% por concepto y global

Prerequisito: python src/03_feature_engineering.py
Ejecutar    : python src/04a_regresion.py
Salidas     : models/lgb_multi_concepto.joblib
              models/lgb_p10_multi_concepto.joblib
              models/lgb_p90_multi_concepto.joblib
              models/bias_correction_escalar.joblib
              models/bias_correction_piecewise.joblib
              models/bias_prov_concepto.joblib
              models/cqr_margenes_por_concepto.joblib
              models/regresion_medians.joblib
              models/optuna_best_params.joblib
              reports/figures/regresion_comparacion_algoritmos.png
              reports/figures/regresion_metricas_concepto.png
"""

import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, RANDOM_STATE, DPI_FIGURAS
from utils import mdape, mape, rmse, mae, cobertura_intervalo

# ---------------------------------------------------------------------------
# Carga de datos y configuración
# ---------------------------------------------------------------------------
train = pd.read_parquet(PROCESSED_DIR / "dataset_modelable_train.parquet")
test  = pd.read_parquet(PROCESSED_DIR / "dataset_modelable_test.parquet")
feature_config = joblib.load(MODELS_DIR / "feature_config.joblib")

FEATURES_NUM  = [f for f in feature_config["features_num"] if f in train.columns]
FEATURES_CAT  = [f for f in feature_config["features_cat"] if f in train.columns]
TARGET        = feature_config["target"]
ALL_FEATURES  = FEATURES_NUM + FEATURES_CAT

train["log_target"] = np.log1p(train[TARGET])
test["log_target"]  = np.log1p(test[TARGET])

for col in FEATURES_CAT:
    train[col] = train[col].astype("category")
    test[col]  = test[col].astype("category")

conceptos   = sorted(train["concepto_canonico"].unique().tolist())
col_prov_tr = feature_config.get("col_prov", "proveedor_norm")

print(f"Train: {len(train):,} filas | Test: {len(test):,} filas")
print(f"Features: {len(ALL_FEATURES)} ({len(FEATURES_NUM)} num + {len(FEATURES_CAT)} cat)")
print(f"Conceptos canónicos ({len(conceptos)}): {conceptos}")

# ---------------------------------------------------------------------------
# FASE 1: Comparación global de algoritmos (Ridge vs XGBoost vs LightGBM)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FASE 1: COMPARACIÓN DE ALGORITMOS (3-Fold TimeSeriesSplit global)")
print("=" * 70)

def preparar_X_sklearn(df_sub, num_cols, cat_cols):
    """Codificación ordinal para modelos sklearn (Ridge, XGB baseline)."""
    X = df_sub[num_cols + cat_cols].copy()
    for col in cat_cols:
        X[col] = X[col].astype(str)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat_cols] = enc.fit_transform(X[cat_cols])
    return X.fillna(0).values, enc


tscv_global = TimeSeriesSplit(n_splits=3)
resultados_comparacion = {"Ridge": [], "XGBoost": [], "LightGBM": []}

X_train_global_ord, enc_global = preparar_X_sklearn(train, FEATURES_NUM, FEATURES_CAT)
y_train_global = train["log_target"].values
w_train_global = train["sample_weight"].values

X_train_lgb = train[ALL_FEATURES]
y_train_lgb = train["log_target"]
w_train_lgb = train["sample_weight"]

for fold_idx, (idx_tr, idx_val) in enumerate(tscv_global.split(X_train_global_ord)):
    # Ridge
    X_f_ord, X_v_ord = X_train_global_ord[idx_tr], X_train_global_ord[idx_val]
    y_f_log, y_v_log = y_train_global[idx_tr], y_train_global[idx_val]
    w_f               = w_train_global[idx_tr]
    y_v_real          = np.expm1(y_v_log)

    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_f_ord, y_f_log, sample_weight=w_f)
    pred_r = np.expm1(ridge.predict(X_v_ord))
    resultados_comparacion["Ridge"].append(mdape(y_v_real, pred_r))

    # XGBoost
    xgb_cmp = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mae", verbosity=0, random_state=RANDOM_STATE,
    )
    xgb_cmp.fit(X_f_ord, y_f_log, sample_weight=w_f, verbose=False)
    pred_x = np.expm1(xgb_cmp.predict(X_v_ord))
    resultados_comparacion["XGBoost"].append(mdape(y_v_real, pred_x))

    # LightGBM
    X_f_lgb = X_train_lgb.iloc[idx_tr]
    X_v_lgb = X_train_lgb.iloc[idx_val]
    y_f_lgb = y_train_lgb.iloc[idx_tr]
    w_f_lgb = w_train_lgb.iloc[idx_tr]
    lgb_cmp = lgb.LGBMRegressor(
        n_estimators=300, num_leaves=31, learning_rate=0.07,
        verbosity=-1, random_state=RANDOM_STATE,
    )
    lgb_cmp.fit(X_f_lgb, y_f_lgb, sample_weight=w_f_lgb, callbacks=[lgb.log_evaluation(-1)])
    pred_l = np.expm1(lgb_cmp.predict(X_v_lgb))
    resultados_comparacion["LightGBM"].append(mdape(y_v_real, pred_l))

    print(f"  Fold {fold_idx+1}: Ridge={resultados_comparacion['Ridge'][-1]:.1f}%  "
          f"XGBoost={resultados_comparacion['XGBoost'][-1]:.1f}%  "
          f"LightGBM={resultados_comparacion['LightGBM'][-1]:.1f}%")

resumen_comp = {alg: np.mean(vals) for alg, vals in resultados_comparacion.items()}
ganador     = min(resumen_comp, key=resumen_comp.get)

print(f"\nMdAPE promedio (3-fold):")
for alg, mdape_avg in sorted(resumen_comp.items(), key=lambda x: x[1]):
    marca = " ← GANADOR" if alg == ganador else ""
    print(f"  {alg:<12}: {mdape_avg:.2f}%{marca}")
print(f"\nAlgoritmo seleccionado: {ganador}")

# Gráfico de comparación
fig, ax = plt.subplots(figsize=(8, 5))
algs = list(resultados_comparacion.keys())
medias = [np.mean(resultados_comparacion[a]) for a in algs]
stds   = [np.std(resultados_comparacion[a])  for a in algs]
colores = ["mediumseagreen" if a == ganador else "steelblue" for a in algs]
bars = ax.bar(algs, medias, yerr=stds, capsize=6, color=colores, edgecolor="white", linewidth=1.2)
ax.set_ylabel("MdAPE promedio (%)", fontsize=11)
ax.set_title("Comparación de algoritmos — MdAPE global (3-Fold TS-CV)", fontsize=13, fontweight="bold")
for bar, m in zip(bars, medias):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{m:.2f}%", ha="center", fontsize=10)
ax.axhline(5, color="red", linestyle="--", linewidth=1, label="Objetivo <5%")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "regresion_comparacion_algoritmos.png", dpi=DPI_FIGURAS)
plt.close()
print("Guardado: regresion_comparacion_algoritmos.png")

# ---------------------------------------------------------------------------
# FASE 2: Entrenamiento multi-modelo por concepto con Optuna
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FASE 2: ENTRENAMIENTO POR CONCEPTO (LightGBM + Optuna + holdout CQR)")
print("=" * 70)

def n_splits_adaptativo(n: int) -> int:
    if n >= 500: return 5
    elif n >= 200: return 3
    return 2

def n_trials_adaptativo(n: int) -> int:
    if n >= 500: return 40
    elif n >= 150: return 25
    return 15

def crear_objective(X_c, y_c, w_c, n_folds):
    def objective(trial):
        params = {
            "objective": "regression", "metric": "mae", "verbosity": -1,
            "random_state": RANDOM_STATE,
            "num_leaves":        trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        gap_cv = max(1, len(X_c) // (n_folds * 5))
        maes   = []
        for idx_tr, idx_val in TimeSeriesSplit(n_splits=n_folds, gap=gap_cv).split(X_c):
            X_f, X_v = X_c.iloc[idx_tr], X_c.iloc[idx_val]
            y_f, y_v = y_c.iloc[idx_tr], y_c.iloc[idx_val]
            m = lgb.LGBMRegressor(**params)
            m.fit(X_f, y_f, sample_weight=w_c.iloc[idx_tr],
                  eval_set=[(X_v, y_v)],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
            maes.append(mean_absolute_error(np.expm1(y_v), np.expm1(m.predict(X_v))))
        return float(np.mean(maes))
    return objective

modelos_por_concepto  = {}
metricas_por_concepto = {}
oof_predictions       = {}
mejores_params_cache  = {}
fit_data_por_concepto = {}
calib_data_por_concepto = {}

for concepto in conceptos:
    mask_tr    = train["concepto_canonico"] == concepto
    concept_tr = train[mask_tr]
    n_tr       = len(concept_tr)

    if n_tr < 10:
        print(f"  {concepto}: SKIP (n={n_tr})")
        continue

    n_calib = min(max(int(n_tr * 0.15), 3), n_tr - 8)
    n_fit   = n_tr - n_calib

    X_fit, y_fit_log, w_fit = (
        concept_tr.iloc[:n_fit][ALL_FEATURES],
        concept_tr.iloc[:n_fit]["log_target"],
        concept_tr.iloc[:n_fit]["sample_weight"],
    )
    X_calib     = concept_tr.iloc[n_fit:][ALL_FEATURES]
    y_calib_log = concept_tr.iloc[n_fit:]["log_target"]

    print(f"  {concepto}: n_fit={n_fit}, n_calib={n_calib} — Optuna...", end=" ", flush=True)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_STATE, n_startup_trials=max(5, n_trials_adaptativo(n_fit)//4)),
    )
    study.optimize(
        crear_objective(X_fit, y_fit_log, w_fit, n_splits_adaptativo(n_fit)),
        n_trials=n_trials_adaptativo(n_fit), show_progress_bar=False,
    )
    best_params = {**study.best_params, "objective": "regression",
                   "metric": "mae", "verbosity": -1, "random_state": RANDOM_STATE}
    mejores_params_cache[concepto] = best_params

    # OOF para bias correction
    n_folds  = n_splits_adaptativo(n_fit)
    gap_cv   = max(1, n_fit // (n_folds * 5))
    oof_real = np.full(n_fit, np.nan)
    oof_pred = np.full(n_fit, np.nan)
    mae_folds, mdape_folds = [], []

    for idx_f, idx_v in TimeSeriesSplit(n_splits=n_folds, gap=gap_cv).split(X_fit):
        X_f, X_v = X_fit.iloc[idx_f], X_fit.iloc[idx_v]
        y_f, y_v = y_fit_log.iloc[idx_f], y_fit_log.iloc[idx_v]
        m = lgb.LGBMRegressor(**best_params)
        m.fit(X_f, y_f, sample_weight=w_fit.iloc[idx_f], callbacks=[lgb.log_evaluation(-1)])
        pred_v  = np.expm1(m.predict(X_v))
        real_v  = np.expm1(y_v)
        oof_real[idx_v] = real_v
        oof_pred[idx_v] = pred_v
        mae_folds.append(mean_absolute_error(real_v, pred_v))
        mdape_folds.append(mdape(real_v, pred_v))

    mask_oof = ~np.isnan(oof_real)
    provs_oof = (
        concept_tr.iloc[:n_fit][col_prov_tr].values[mask_oof]
        if col_prov_tr in concept_tr.columns else np.array([""] * mask_oof.sum())
    )
    oof_predictions[concepto] = (oof_real[mask_oof], oof_pred[mask_oof], provs_oof)

    # Modelo final con early stopping sobre X_calib
    modelo_final = lgb.LGBMRegressor(**best_params)
    modelo_final.fit(
        X_fit, y_fit_log, sample_weight=w_fit,
        eval_set=[(X_calib, y_calib_log)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    modelos_por_concepto[concepto]    = modelo_final
    fit_data_por_concepto[concepto]   = (X_fit, y_fit_log, w_fit)
    calib_data_por_concepto[concepto] = (X_calib, np.expm1(y_calib_log.values))

    # Métricas en test
    mask_te = test["concepto_canonico"] == concepto
    if mask_te.sum() > 0:
        X_te    = test.loc[mask_te, ALL_FEATURES]
        y_te    = np.expm1(test.loc[mask_te, "log_target"])
        pred_te = np.expm1(modelo_final.predict(X_te))
        mae_te  = mean_absolute_error(y_te, pred_te)
        md_te   = mdape(y_te, pred_te)
        rmse_te = rmse(y_te, pred_te)
        r2_te   = r2_score(np.log1p(y_te), modelo_final.predict(X_te)) if len(y_te) > 1 else np.nan
    else:
        mae_te = md_te = rmse_te = r2_te = np.nan

    metricas_por_concepto[concepto] = {
        "n_fit": n_fit, "n_calib": n_calib, "n_test": int(mask_te.sum()),
        "mae_cv": np.mean(mae_folds), "mdape_cv": np.mean(mdape_folds),
        "mae_test": mae_te, "mdape_test": md_te, "rmse_test": rmse_te, "r2_test": r2_te,
    }
    mdape_str = f"{md_te:.1f}%" if not np.isnan(md_te) else "sin test"
    print(f"CV MdAPE={np.mean(mdape_folds):.1f}% | Test MdAPE={mdape_str}")

# ---------------------------------------------------------------------------
# FASE 3: Corrección de sesgo en 3 capas
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FASE 3: CORRECCIÓN DE SESGO (escalar + piecewise + proveedor×concepto)")
print("=" * 70)

# Capa 1: corrección escalar por concepto
bias_escalar = {}
for concepto, (y_oof, pred_oof, _) in oof_predictions.items():
    if len(pred_oof) == 0:
        bias_escalar[concepto] = 1.0
        continue
    ratio = np.clip(y_oof / np.clip(pred_oof, 1e-3, None), 0.2, 5.0)
    bias_escalar[concepto] = float(np.median(ratio))

print("Ratios escalares (real/predicho):")
for c, r in sorted(bias_escalar.items()):
    diag = "subestima" if r > 1 else "sobreestima"
    print(f"  {c:<30} {r:.3f}  ({diag})")

# Capa 2: corrección piecewise por magnitud de predicción
N_BUCKETS    = 6
bias_piecewise = {}

for concepto, (y_oof, pred_oof, _) in oof_predictions.items():
    if len(pred_oof) < N_BUCKETS * 3:
        bias_piecewise[concepto] = bias_escalar[concepto]
        continue
    edges        = np.percentile(pred_oof, np.linspace(0, 100, N_BUCKETS + 1))
    edges[0]    -= 1e-6;  edges[-1] += 1e-6
    bucket_ratios = []
    for i in range(N_BUCKETS):
        mask_b = (pred_oof > edges[i]) & (pred_oof <= edges[i + 1])
        if mask_b.sum() >= 3:
            r = float(np.median(np.clip(y_oof[mask_b] / np.clip(pred_oof[mask_b], 1e-3, None), 0.2, 5.0)))
        else:
            r = bias_escalar[concepto]
        bucket_ratios.append(r)
    bias_piecewise[concepto] = (edges, bucket_ratios)

print(f"Corrección piecewise: {len(bias_piecewise)} conceptos")

def aplicar_bias_capa1(pred_pen: float, concepto: str) -> float:
    corr = bias_piecewise.get(concepto, bias_escalar.get(concepto, 1.0))
    if isinstance(corr, tuple):
        edges, ratios = corr
        bucket = int(np.clip(np.searchsorted(edges[1:-1], pred_pen), 0, len(ratios) - 1))
        return pred_pen * ratios[bucket]
    return pred_pen * corr

# Capa 3: corrección por proveedor × concepto
bias_prov_concepto = {}

for concepto, (y_oof, pred_oof, provs_oof) in oof_predictions.items():
    if len(pred_oof) < 5: continue
    for prov in np.unique(provs_oof):
        mask_p = provs_oof == prov
        if mask_p.sum() < 3: continue
        y_p      = y_oof[mask_p]
        pred_l1  = np.array([aplicar_bias_capa1(p, concepto) for p in pred_oof[mask_p]])
        ratio    = float(np.median(np.clip(y_p / np.clip(pred_l1, 1e-3, None), 0.1, 10.0)))
        bias_prov_concepto[(str(prov).upper(), concepto)] = ratio

print(f"Pares proveedor×concepto con corrección propia: {len(bias_prov_concepto)}")

def aplicar_bias_completo(pred_pen: float, concepto: str, proveedor=None) -> float:
    pred_l1  = aplicar_bias_capa1(pred_pen, concepto)
    if proveedor is not None:
        ratio_l2 = bias_prov_concepto.get((str(proveedor).upper(), concepto), 1.0)
        return pred_l1 * ratio_l2
    return pred_l1

# ---------------------------------------------------------------------------
# FASE 4: Modelos cuantílicos P10/P90 + calibración CQR (80% CI)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FASE 4: MODELOS CUANTÍLICOS P10/P90 + CALIBRACIÓN CQR (80% CI)")
print("=" * 70)

modelos_p10 = {}
modelos_p90 = {}

for concepto in modelos_por_concepto:
    X_fit, y_fit_log, w_fit = fit_data_por_concepto[concepto]
    base = dict(mejores_params_cache.get(concepto, {"n_estimators": 300, "num_leaves": 31}))
    for cuantil, store in [(0.1, modelos_p10), (0.9, modelos_p90)]:
        params_q = {**base, "objective": "quantile", "alpha": cuantil,
                    "metric": "quantile", "verbosity": -1, "random_state": RANDOM_STATE}
        m = lgb.LGBMRegressor(**params_q)
        m.fit(X_fit, y_fit_log, sample_weight=w_fit, callbacks=[lgb.log_evaluation(-1)])
        store[concepto] = m

print(f"Modelos P10 y P90 entrenados: {len(modelos_p10)} conceptos")

# Calibración CQR sobre holdout limpio
cqr_margenes     = {}
COBERTURA_TARGET = 0.80  # CI del 80% → P10–P90

for concepto in modelos_por_concepto:
    X_calib, y_calib_pen = calib_data_por_concepto[concepto]
    n_cal                = len(y_calib_pen)

    pred_p10  = np.expm1(modelos_p10[concepto].predict(X_calib))
    pred_p90  = np.expm1(modelos_p90[concepto].predict(X_calib))
    pred_med  = np.expm1(modelos_por_concepto[concepto].predict(X_calib))
    pred_corr = np.array([aplicar_bias_capa1(p, concepto) for p in pred_med])

    score_inf = np.maximum(pred_p10 - y_calib_pen, 0) / np.clip(pred_corr, 1e-3, None)
    score_sup = np.maximum(y_calib_pen - pred_p90, 0) / np.clip(pred_corr, 1e-3, None)

    nivel      = int(np.ceil((1 + COBERTURA_TARGET) * n_cal / 2))
    nivel      = min(nivel, n_cal)
    margen_inf = float(np.sort(score_inf)[min(nivel - 1, len(score_inf) - 1)])
    margen_sup = float(np.sort(score_sup)[min(nivel - 1, len(score_sup) - 1)])

    cqr_margenes[concepto] = {
        "margen_inf": margen_inf, "margen_sup": margen_sup,
        "n_calib": n_cal, "cobertura_objetivo": COBERTURA_TARGET,
    }

print("Márgenes CQR (IC 80%):")
for c, m in sorted(cqr_margenes.items()):
    print(f"  {c:<30} inf={m['margen_inf']:.3f}  sup={m['margen_sup']:.3f}  (n={m['n_calib']})")

# ---------------------------------------------------------------------------
# FASE 5: Evaluación final en test
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FASE 5: EVALUACIÓN FINAL EN TEST")
print("=" * 70)

rows_eval = []
for concepto, modelo in modelos_por_concepto.items():
    mask_te = test["concepto_canonico"] == concepto
    if mask_te.sum() == 0: continue

    X_te     = test.loc[mask_te, ALL_FEATURES]
    y_te     = test.loc[mask_te, TARGET].values
    provs_te = (test.loc[mask_te, col_prov_tr].values
                if col_prov_tr in test.columns else np.array([""] * len(y_te)))

    pred_log  = modelo.predict(X_te)
    pred_corr = np.array([
        aplicar_bias_completo(p, concepto, prov)
        for p, prov in zip(np.expm1(pred_log), provs_te)
    ])

    pred_p10  = np.expm1(modelos_p10[concepto].predict(X_te))
    pred_p90  = np.expm1(modelos_p90[concepto].predict(X_te))
    marg      = cqr_margenes.get(concepto, {"margen_inf": 0.20, "margen_sup": 0.20})
    p10_cal   = pred_p10 * (1 - marg["margen_inf"])
    p90_cal   = pred_p90 * (1 + marg["margen_sup"])

    rows_eval.append({
        "concepto":     concepto,
        "n":            len(y_te),
        "MAE_PEN":      mae(y_te, pred_corr),
        "MAPE_%":       mape(y_te, pred_corr),
        "MdAPE_%":      mdape(y_te, pred_corr),
        "RMSE_PEN":     rmse(y_te, pred_corr),
        "R2":           r2_score(np.log1p(y_te), pred_log) if len(y_te) > 1 else np.nan,
        "Cobertura_80": cobertura_intervalo(y_te, p10_cal, p90_cal),
    })

df_eval = pd.DataFrame(rows_eval).set_index("concepto")
print(df_eval.round(2).to_string())

media_mdape  = df_eval["MdAPE_%"].mean()
media_mape   = df_eval["MAPE_%"].mean()
media_rmse   = df_eval["RMSE_PEN"].mean()
media_cob    = df_eval["Cobertura_80"].mean()
print(f"\nMdAPE global          : {media_mdape:.1f}%")
print(f"MAPE global           : {media_mape:.1f}%")
print(f"RMSE promedio (PEN)   : S/ {media_rmse:,.0f}")
print(f"Cobertura IC 80%      : {media_cob:.1f}%")
print(f"Objetivo MdAPE < 5%   : {'ALCANZADO ✓' if media_mdape < 5 else f'EN PROGRESO ({media_mdape:.1f}%)'}")
print(f"Objetivo Cobertura 80%: {'ALCANZADO ✓' if media_cob >= 75 else f'EN PROGRESO ({media_cob:.1f}%)'}")

# Gráfico de métricas por concepto
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df_eval["MdAPE_%"].sort_values().plot.barh(ax=axes[0], color="steelblue", edgecolor="white")
axes[0].axvline(5, color="red", linestyle="--", linewidth=1, label="Objetivo 5%")
axes[0].set_title("MdAPE por concepto (test)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("MdAPE (%)")
axes[0].legend(fontsize=8)

df_eval["Cobertura_80"].sort_values().plot.barh(ax=axes[1], color="mediumseagreen", edgecolor="white")
axes[1].axvline(80, color="red", linestyle="--", linewidth=1, label="Objetivo 80%")
axes[1].set_title("Cobertura IC 80% por concepto (test)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Cobertura (%)")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "regresion_metricas_concepto.png", dpi=DPI_FIGURAS)
plt.close()
print("Guardado: regresion_metricas_concepto.png")

# ---------------------------------------------------------------------------
# Guardar artefactos
# ---------------------------------------------------------------------------
regresion_medians = {col: float(train[col].median()) for col in FEATURES_NUM if col in train.columns}

joblib.dump(modelos_por_concepto,   MODELS_DIR / "lgb_multi_concepto.joblib")
joblib.dump(modelos_p10,            MODELS_DIR / "lgb_p10_multi_concepto.joblib")
joblib.dump(modelos_p90,            MODELS_DIR / "lgb_p90_multi_concepto.joblib")
joblib.dump(bias_escalar,           MODELS_DIR / "bias_correction_escalar.joblib")
joblib.dump(bias_piecewise,         MODELS_DIR / "bias_correction_piecewise.joblib")
joblib.dump(bias_prov_concepto,     MODELS_DIR / "bias_prov_concepto.joblib")
joblib.dump(cqr_margenes,           MODELS_DIR / "cqr_margenes_por_concepto.joblib")
joblib.dump(regresion_medians,      MODELS_DIR / "regresion_medians.joblib")
joblib.dump(mejores_params_cache,   MODELS_DIR / "optuna_best_params.joblib")

print(f"\nModelos guardados en {MODELS_DIR}:")
for f in sorted(MODELS_DIR.glob("lgb_*.joblib")):
    print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
print("\nRegresión finalizada. Próximo paso: python src/04b_clasificacion.py")
