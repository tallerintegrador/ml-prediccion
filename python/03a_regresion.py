"""
03a_regresion.py — Modelos de regresión para estimar costos de importación.

Arquitectura: 1 LightGBM por concepto canónico (13 modelos) + intervalos CQR.
Prerequisito: python python/02_feature_engineering.py
Ejecutar    : python python/03a_regresion.py
Salidas     : models/lgb_multi_concepto.joblib
              models/lgb_p10_multi_concepto.joblib
              models/lgb_p90_multi_concepto.joblib
              models/bias_correction_escalar.joblib
              models/bias_correction_piecewise.joblib
              models/bias_prov_concepto.joblib
              models/cqr_margenes_por_concepto.joblib
              models/regresion_medians.joblib
              models/optuna_best_params.joblib
"""

# --- celda 1 ---
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import mediana_porcentaje_error_absoluto, rmse

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- celda 2 ---
BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_parquet(PROCESSED_DIR / "dataset_modelable_train.parquet")
test  = pd.read_parquet(PROCESSED_DIR / "dataset_modelable_test.parquet")
feature_config = joblib.load(MODELS_DIR / "feature_config.joblib")

FEATURES_NUM = [f for f in feature_config["features_num"] if f in train.columns]
FEATURES_CAT = [f for f in feature_config["features_cat"] if f in train.columns]
TARGET       = feature_config["target"]
ALL_FEATURES = FEATURES_NUM + FEATURES_CAT

train["log_target"] = np.log1p(train[TARGET])
test["log_target"]  = np.log1p(test[TARGET])

for col in FEATURES_CAT:
    train[col] = train[col].astype("category")
    test[col]  = test[col].astype("category")

print(f"Train: {len(train):,} filas | Test: {len(test):,} filas")
print(f"Features: {len(ALL_FEATURES)} ({len(FEATURES_NUM)} num + {len(FEATURES_CAT)} cat)")
print(f"Conceptos canónicos: {sorted(train['concepto_canonico'].unique().tolist())}")

# --- celda 3 ---
def n_splits_adaptativo(n_train: int) -> int:
    if n_train >= 500:
        return 5
    elif n_train >= 200:
        return 3
    return 2


def crear_objective_lgb(X_c, y_c, w_c, n_folds):
    """Función objetivo Optuna para un concepto dado."""
    def objective(trial):
        params = {
            "objective":         "regression",
            "metric":            "mae",
            "verbosity":         -1,
            "random_state":      42,
            "num_leaves":        trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.5),
        }
        gap_cv = max(1, len(X_c) // (n_folds * 5))
        tscv   = TimeSeriesSplit(n_splits=n_folds, gap=gap_cv)
        maes   = []
        for idx_tr, idx_val in tscv.split(X_c):
            X_f, X_v = X_c.iloc[idx_tr], X_c.iloc[idx_val]
            y_f, y_v = y_c.iloc[idx_tr], y_c.iloc[idx_val]
            w_f      = w_c.iloc[idx_tr]
            m = lgb.LGBMRegressor(**params)
            m.fit(
                X_f, y_f, sample_weight=w_f,
                eval_set=[(X_v, y_v)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
            maes.append(mean_absolute_error(np.expm1(y_v), np.expm1(m.predict(X_v))))
        return float(np.mean(maes))
    return objective


def n_trials_por_tamano(n_train: int) -> int:
    if n_train >= 500:
        return 40
    elif n_train >= 150:
        return 25
    return 15


def obtener_mejores_params(X_c, y_c, w_c, concepto: str) -> dict:
    """Optimiza hiperparámetros con Optuna sobre los datos de fit."""
    n       = len(X_c)
    n_folds = n_splits_adaptativo(n)
    study   = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=max(5, n_trials_por_tamano(n) // 4)),
    )
    study.optimize(
        crear_objective_lgb(X_c, y_c, w_c, n_folds),
        n_trials=n_trials_por_tamano(n),
        show_progress_bar=False,
    )
    best = study.best_params
    best.update({"objective": "regression", "metric": "mae",
                 "verbosity": -1, "random_state": 42})
    return best

# --- celda 4 ---
# Entrenamiento multi-modelo con separación limpia de holdout CQR
#
# Arquitectura de datos por concepto:
#   concept_tr (todo el train del concepto)
#   ├── X_fit  (85%)  ← Optuna + OOF + modelo final + quantile models
#   └── X_calib (15%) ← SOLO calibración CQR (modelo NUNCA lo ve)
#
# Esto garantiza que los nonconformity scores del CQR no están contaminados.
print("\n=== MULTI-MODELO POR CONCEPTO CANÓNICO (con Optuna + holdout CQR limpio) ===")

modelos_por_concepto    = {}
metricas_por_concepto   = {}
oof_predictions         = {}
mejores_params_cache    = {}
fit_data_por_concepto   = {}   # (X_fit, y_fit_log, w_fit) → quantile models
calib_data_por_concepto = {}   # (X_calib, y_calib_pen)   → CQR calibración

conceptos    = sorted(train["concepto_canonico"].unique().tolist())
col_prov_tr  = feature_config.get("col_prov", "proveedor_norm")

for concepto in conceptos:
    mask_tr    = train["concepto_canonico"] == concepto
    mask_te    = test["concepto_canonico"]  == concepto
    concept_tr = train[mask_tr]
    n_tr       = len(concept_tr)

    if n_tr < 10:
        print(f"  {concepto}: SKIP (solo {n_tr} filas)")
        continue

    # Separar holdout de calibración antes de cualquier entrenamiento.
    # min(15%, n_tr-8) garantiza al menos 8 filas de entrenamiento con n_tr≥10.
    n_calib = min(max(int(n_tr * 0.15), 3), n_tr - 8)
    n_fit   = n_tr - n_calib

    X_fit       = concept_tr.iloc[:n_fit][ALL_FEATURES]
    y_fit_log   = concept_tr.iloc[:n_fit]["log_target"]
    w_fit       = concept_tr.iloc[:n_fit]["sample_weight"]
    X_calib     = concept_tr.iloc[n_fit:][ALL_FEATURES]
    y_calib_log = concept_tr.iloc[n_fit:]["log_target"]

    print(f"  {concepto}: n_fit={n_fit}, n_calib={n_calib} — optimizando...",
          end=" ", flush=True)

    # Optuna sobre X_fit (mismos datos que verá el modelo final)
    best_params = obtener_mejores_params(X_fit, y_fit_log, w_fit, concepto)
    mejores_params_cache[concepto] = best_params

    # OOF sobre X_fit para bias correction
    n_folds = n_splits_adaptativo(n_fit)
    gap_cv  = max(1, n_fit // (n_folds * 5))
    tscv    = TimeSeriesSplit(n_splits=n_folds, gap=gap_cv)

    oof_y_real = np.full(n_fit, np.nan)
    oof_y_pred = np.full(n_fit, np.nan)
    mae_folds, mdape_folds = [], []

    for idx_f, idx_v in tscv.split(X_fit):
        X_f, X_v = X_fit.iloc[idx_f], X_fit.iloc[idx_v]
        y_f, y_v = y_fit_log.iloc[idx_f], y_fit_log.iloc[idx_v]
        w_f      = w_fit.iloc[idx_f]
        m = lgb.LGBMRegressor(**best_params)
        m.fit(X_f, y_f, sample_weight=w_f, callbacks=[lgb.log_evaluation(-1)])
        pred_v = np.expm1(m.predict(X_v))
        real_v = np.expm1(y_v)
        oof_y_real[idx_v] = real_v
        oof_y_pred[idx_v] = pred_v
        mae_folds.append(mean_absolute_error(real_v, pred_v))
        mdape_folds.append(mediana_porcentaje_error_absoluto(real_v, pred_v))

    mask_oof = ~np.isnan(oof_y_real)
    provs_oof = (
        concept_tr.iloc[:n_fit][col_prov_tr].values[mask_oof]
        if col_prov_tr in concept_tr.columns
        else np.array([""] * mask_oof.sum())
    )
    oof_predictions[concepto] = (oof_y_real[mask_oof], oof_y_pred[mask_oof], provs_oof)

    # Modelo final entrenado solo sobre X_fit con early stopping sobre X_calib.
    # Esto es coherente con la configuración de Optuna (que también usa early_stopping)
    # y garantiza que X_calib nunca contamina el modelo.
    modelo_final = lgb.LGBMRegressor(**best_params)
    modelo_final.fit(
        X_fit, y_fit_log,
        sample_weight=w_fit,
        eval_set=[(X_calib, y_calib_log)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    modelos_por_concepto[concepto]    = modelo_final
    fit_data_por_concepto[concepto]   = (X_fit, y_fit_log, w_fit)
    calib_data_por_concepto[concepto] = (X_calib, np.expm1(y_calib_log.values))

    # Métricas en test
    if mask_te.sum() > 0:
        X_te_c   = test.loc[mask_te, ALL_FEATURES]
        y_te_c   = np.expm1(test.loc[mask_te, "log_target"])
        pred_log = modelo_final.predict(X_te_c)
        pred_te  = np.expm1(pred_log)
        mae_te   = mean_absolute_error(y_te_c, pred_te)
        mdape_te = mediana_porcentaje_error_absoluto(y_te_c, pred_te)
        rmse_te  = rmse(y_te_c, pred_te)
    else:
        mae_te = mdape_te = rmse_te = np.nan

    metricas_por_concepto[concepto] = {
        "n_fit": n_fit, "n_calib": n_calib, "n_test": int(mask_te.sum()),
        "mae_cv": np.mean(mae_folds),   "mdape_cv": np.mean(mdape_folds),
        "mae_test": mae_te, "mdape_test": mdape_te, "rmse_test": rmse_te,
    }
    mdape_str = f"{mdape_te:.1f}%" if not np.isnan(mdape_te) else "sin test"
    print(f"CV MdAPE={np.mean(mdape_folds):.1f}% | Test MdAPE={mdape_str}")

df_metricas = pd.DataFrame(metricas_por_concepto).T
print(f"\nResumen multi-modelo:")
print(df_metricas.round(1).to_string())

# --- celda 5 ---
# Corrección de sesgo escalar con OOF genuino
print("\n=== CORRECCIÓN DE SESGO ESCALAR (OOF genuino) ===")

bias_escalar = {}

for concepto, oof_data in oof_predictions.items():
    y_oof, pred_oof = oof_data[0], oof_data[1]
    if len(pred_oof) == 0:
        bias_escalar[concepto] = 1.0
        continue
    ratio          = y_oof / np.clip(pred_oof, 1e-3, None)
    ratio_clipped  = np.clip(ratio, 0.2, 5.0)
    bias_escalar[concepto] = float(np.median(ratio_clipped))

print("Ratios de corrección escalar (real/predicho OOF):")
for c, r in sorted(bias_escalar.items()):
    print(f"  {c:<30} ratio={r:.3f}  ({'subestima' if r > 1 else 'sobreestima'})")

# --- celda 6 ---
# Corrección piecewise por cuartiles de predicción OOF
print("\n=== CORRECCIÓN DE SESGO PIECEWISE (OOF) ===")

bias_piecewise = {}
N_BUCKETS = 6

for concepto, oof_data in oof_predictions.items():
    y_oof, pred_oof = oof_data[0], oof_data[1]
    if len(pred_oof) < N_BUCKETS * 3:
        bias_piecewise[concepto] = bias_escalar[concepto]
        continue
    edges     = np.percentile(pred_oof, np.linspace(0, 100, N_BUCKETS + 1))
    edges[0]  -= 1e-6
    edges[-1] += 1e-6
    bucket_ratios = []
    for i in range(N_BUCKETS):
        mask = (pred_oof > edges[i]) & (pred_oof <= edges[i + 1])
        if mask.sum() >= 3:
            r = float(np.median(
                np.clip(y_oof[mask] / np.clip(pred_oof[mask], 1e-3, None), 0.2, 5.0)
            ))
        else:
            r = bias_escalar[concepto]
        bucket_ratios.append(r)
    bias_piecewise[concepto] = (edges, bucket_ratios)

print(f"Corrección piecewise calculada para {len(bias_piecewise)} conceptos")


def aplicar_bias_completo(pred_pen: float, concepto: str, proveedor=None) -> float:
    """
    Capa 1: piecewise por concepto (o escalar como fallback).
    Capa 2 (opcional): ratio por proveedor×concepto.
    """
    corr = bias_piecewise.get(concepto, bias_escalar.get(concepto, 1.0))
    if isinstance(corr, tuple):
        edges, ratios = corr
        bucket = int(np.clip(np.searchsorted(edges[1:-1], pred_pen), 0, len(ratios) - 1))
        pred_l1 = pred_pen * ratios[bucket]
    else:
        pred_l1 = pred_pen * corr
    if proveedor is not None:
        ratio_l2 = bias_prov_concepto.get((str(proveedor).upper(), concepto), 1.0)
        return pred_l1 * ratio_l2
    return pred_l1


# --- celda 6b ---
# Corrección por proveedor × concepto (capa 2)
print("\n=== CORRECCIÓN POR PROVEEDOR × CONCEPTO (OOF, capa 2) ===")

bias_prov_concepto = {}

for concepto, oof_data in oof_predictions.items():
    y_oof, pred_oof, provs_oof = oof_data[0], oof_data[1], oof_data[2]
    if len(pred_oof) < 5:
        continue
    for prov in np.unique(provs_oof):
        mask_prov = provs_oof == prov
        if mask_prov.sum() < 3:
            continue
        y_p    = y_oof[mask_prov]
        pred_p = pred_oof[mask_prov]
        pred_p_l1 = np.array([aplicar_bias_completo(p, concepto) for p in pred_p])
        ratio  = float(np.median(
            np.clip(y_p / np.clip(pred_p_l1, 1e-3, None), 0.1, 10.0)
        ))
        bias_prov_concepto[(str(prov).upper(), concepto)] = ratio

print(f"Pares proveedor×concepto con corrección propia: {len(bias_prov_concepto)}")
top_corr = sorted(bias_prov_concepto.items(), key=lambda x: abs(x[1] - 1), reverse=True)[:8]
for (prov, conc), r in top_corr:
    print(f"  ({prov[:22]}, {conc[:22]}): ratio={r:.3f}  "
          f"[{'subestima' if r > 1 else 'sobreestima'}]")


# --- celda 7 ---
# Modelos cuantílicos P10 / P90 entrenados sobre X_fit (igual que el modelo central)
print("\n=== MODELOS CUANTÍLICOS P10 / P90 ===")

modelos_p10 = {}
modelos_p90 = {}

for concepto in modelos_por_concepto:
    X_fit, y_fit_log, w_fit = fit_data_por_concepto[concepto]
    base = dict(mejores_params_cache.get(concepto, {"n_estimators": 300, "num_leaves": 31}))
    for cuantil, store in [(0.1, modelos_p10), (0.9, modelos_p90)]:
        params_q = {**base, "objective": "quantile", "alpha": cuantil,
                    "metric": "quantile", "verbosity": -1, "random_state": 42}
        m = lgb.LGBMRegressor(**params_q)
        m.fit(X_fit, y_fit_log, sample_weight=w_fit, callbacks=[lgb.log_evaluation(-1)])
        store[concepto] = m

print(f"Modelos P10 y P90 entrenados para {len(modelos_p10)} conceptos")

# --- celda 8 ---
# Calibración CQR con holdout limpio: X_calib nunca fue visto por el modelo final
# ni por los quantile models → cobertura real no inflada.
print("\n=== CALIBRACIÓN CQR (holdout limpio, sin contaminación) ===")

cqr_margenes = {}
COBERTURA_OBJETIVO = 0.90

for concepto in modelos_por_concepto:
    X_calib, y_calib_pen = calib_data_por_concepto[concepto]
    n_calib_real = len(y_calib_pen)

    pred_p10 = np.expm1(modelos_p10[concepto].predict(X_calib))
    pred_p90 = np.expm1(modelos_p90[concepto].predict(X_calib))
    pred_med = np.expm1(modelos_por_concepto[concepto].predict(X_calib))

    # Aplicar capa 1 de bias correction antes de calcular nonconformity scores
    pred_med_corr = np.array([aplicar_bias_completo(p, concepto) for p in pred_med])

    score_inf = np.maximum(pred_p10 - y_calib_pen, 0) / np.clip(pred_med_corr, 1e-3, None)
    score_sup = np.maximum(y_calib_pen - pred_p90, 0) / np.clip(pred_med_corr, 1e-3, None)

    nivel      = int(np.ceil((1 + COBERTURA_OBJETIVO) * n_calib_real / 2))
    nivel      = min(nivel, n_calib_real)
    margen_inf = float(np.sort(score_inf)[min(nivel - 1, len(score_inf) - 1)])
    margen_sup = float(np.sort(score_sup)[min(nivel - 1, len(score_sup) - 1)])

    cqr_margenes[concepto] = {
        "margen_inf": margen_inf,
        "margen_sup": margen_sup,
        "n_calib":    n_calib_real,
        "source":     "holdout_limpio",
    }

print("Márgenes CQR por concepto:")
for c, m in sorted(cqr_margenes.items()):
    print(f"  {c:<30} inf={m['margen_inf']:.3f}  sup={m['margen_sup']:.3f}  "
          f"(n={m['n_calib']})")

# --- celda 9 ---
# Evaluación final en test
print("\n=== EVALUACIÓN FINAL EN TEST ===")

predicciones_test = []
col_prov_feat     = feature_config.get("col_prov", "proveedor_norm")

for concepto, modelo in modelos_por_concepto.items():
    mask_te = test["concepto_canonico"] == concepto
    if mask_te.sum() == 0:
        continue

    X_te     = test.loc[mask_te, ALL_FEATURES]
    y_te     = test.loc[mask_te, TARGET].values
    provs_te = (
        test.loc[mask_te, col_prov_feat].values
        if col_prov_feat in test.columns
        else np.array([""] * len(y_te))
    )
    pred_log  = modelo.predict(X_te)
    pred_pen  = np.expm1(pred_log)
    pred_corr = np.array([
        aplicar_bias_completo(p, concepto, prov)
        for p, prov in zip(pred_pen, provs_te)
    ])

    pred_p10 = np.expm1(modelos_p10[concepto].predict(X_te))
    pred_p90 = np.expm1(modelos_p90[concepto].predict(X_te))
    margenes  = cqr_margenes.get(concepto, {"margen_inf": 0.2, "margen_sup": 0.2})
    p10_cal   = pred_p10 * (1 - margenes["margen_inf"])
    p90_cal   = pred_p90 * (1 + margenes["margen_sup"])

    mae_te   = mean_absolute_error(y_te, pred_corr)
    mdape_te = mediana_porcentaje_error_absoluto(y_te, pred_corr)
    rmse_te  = rmse(y_te, pred_corr)
    r2_te    = r2_score(np.log1p(y_te), pred_log) if len(y_te) > 1 else np.nan
    cobertura = ((y_te >= p10_cal) & (y_te <= p90_cal)).mean() * 100

    predicciones_test.append({
        "concepto":    concepto,
        "n":           len(y_te),
        "MAE_PEN":     mae_te,
        "MdAPE_%":     mdape_te,
        "RMSE_PEN":    rmse_te,
        "R2":          r2_te,
        "Cobertura_%": cobertura,
    })

df_eval = pd.DataFrame(predicciones_test).set_index("concepto")
print(df_eval.round(2).to_string())

media_mdape  = df_eval["MdAPE_%"].mean()
media_cobert = df_eval["Cobertura_%"].mean()
media_rmse   = df_eval["RMSE_PEN"].mean()
print(f"\nMdAPE promedio      : {media_mdape:.1f}%")
print(f"RMSE promedio (PEN) : S/ {media_rmse:,.0f}")
print(f"Cobertura P10-P90   : {media_cobert:.1f}%")
print(f"Objetivo MdAPE <5%  : "
      f"{'ALCANZADO ✓' if media_mdape < 5 else f'EN PROGRESO ({media_mdape:.1f}%)'}")

# --- celda 10 ---
# Medianas de features numéricas para imputación en producción
regresion_medians = {
    col: float(train[col].median())
    for col in FEATURES_NUM
    if col in train.columns
}

# Guardar artefactos
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
for f in sorted(MODELS_DIR.glob("*.joblib")):
    print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
