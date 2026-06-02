# -*- coding: utf-8 -*-
"""
08_modelado.py — Fase de Machine Learning: costos de SERVICIOS logísticos.

Pipeline:
  1. Carga (dataset.load) + features (features.prepare/ColumnTransformer).
  2. Baseline = mediana por segmento (mode x categoria_canon).
  3. Modelos ML: LightGBM (principal), XGBoost, RandomForest. Target = log1p(servicios),
     back-transform con corrección de sesgo (log-normal smearing exp(var/2)).
  4. Validación TEMPORAL forward-chaining por año (train <= t, val t+1). Encoders se
     ajustan SÓLO con el train de cada fold (sin leakage).
  5. Métricas: MAE, RMSE, WAPE, MdAPE (global y por segmento).
  6. Comparación: baseline vs ML; modelo directo vs modelo por concepto.
  7. Intervalos P10/P90 con LightGBM cuantílico (pinball) + cobertura.
  8. Predicción de operaciones pendientes sin facturar (target no fiable).
  9. Artefactos: models/*.pkl, reports/14_modelado.md, reports/*.csv.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import config as C
import dataset as D
import features as F

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

SEED = C.SEED
MODELS_DIR = os.path.join(C.ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
SEG = ["mode", "categoria_canon"]

# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def metrics(y, yhat, ape_floor=50.0):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    wape = float(np.sum(np.abs(err)) / max(np.sum(np.abs(y)), 1e-9))
    mask = y >= ape_floor
    mdape = float(np.median(np.abs(err[mask] / y[mask]))) if mask.sum() else np.nan
    return dict(n=int(len(y)), MAE=mae, RMSE=rmse, WAPE=wape, MdAPE=mdape,
                bias=float(np.mean(err)))


def back_transform(pred_log, log_corr):
    """log1p -> nivel con corrección de sesgo aditiva en log: expm1(pred + log_corr) - clip>=0."""
    return np.clip(np.expm1(pred_log + log_corr), 0, None)


# corrección de sesgo al des-transformar log1p:
#   - "lognormal": paramétrica, 0.5*Var(resid)  (asume residuos ~ Normal en log)
#   - "smearing" : Duan no paramétrica, log(mean(exp(resid)))  (robusta a cola pesada)
CORR_METHOD = "smearing"


def log_correction(resid, method=CORR_METHOD):
    resid = np.asarray(resid, float)
    if method == "lognormal":
        return 0.5 * float(np.var(resid))
    # smearing de Duan ROBUSTO: winsoriza residuos a [p1,p99] antes de exp para que
    # una cola pesada no haga explotar el factor mean(exp(resid)) (visto en el ratio).
    lo, hi = np.percentile(resid, [1, 99])
    return float(np.log(np.mean(np.exp(np.clip(resid, lo, hi)))))


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
# hiperparámetros LightGBM por defecto (sobreescribibles por el tuning de Optuna)
LGBM_PARAMS = dict(objective="regression", n_estimators=500, learning_rate=0.03,
                   num_leaves=31, min_child_samples=20, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                   random_state=SEED, n_jobs=-1, verbose=-1)


def make_model(kind, params=None):
    if kind == "lgbm":
        p = dict(LGBM_PARAMS)
        if params:
            p.update(params)
        m = LGBMRegressor(**p)
    elif kind == "xgb":
        m = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         random_state=SEED, n_jobs=-1, verbosity=0)
    elif kind == "rf":
        m = RandomForestRegressor(n_estimators=400, min_samples_leaf=5,
                                  n_jobs=-1, random_state=SEED)
    else:
        raise ValueError(kind)
    return Pipeline([("ct", F.build_column_transformer()), ("m", m)])


def make_quantile(alpha):
    m = LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=500,
                      learning_rate=0.03, num_leaves=31, min_child_samples=20,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    return Pipeline([("ct", F.build_column_transformer()), ("m", m)])


def conformal_margin(q10, q90, Xcal, ycal_log, alpha=0.20):
    """CQR: margen (en log) que recalibra [q10,q90] a cobertura nominal 1-alpha."""
    lo = q10.predict(Xcal); hi = q90.predict(Xcal)
    s = np.maximum(lo - ycal_log, ycal_log - hi)         # score de no-conformidad
    n = len(s)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(s)[k - 1])


# --------------------------------------------------------------------------- #
# Baseline por segmento
# --------------------------------------------------------------------------- #
def baseline_fit(df):
    g = df.groupby(SEG, observed=True)["target_servicios"].median()
    g_mode = df.groupby("mode", observed=True)["target_servicios"].median()
    glob = float(df["target_servicios"].median())
    return {"seg": g, "mode": g_mode, "glob": glob}


def baseline_predict(model, df):
    out = []
    for _, r in df[SEG].iterrows():
        key = (r["mode"], r["categoria_canon"])
        if key in model["seg"].index:
            out.append(model["seg"].loc[key])
        elif r["mode"] in model["mode"].index:
            out.append(model["mode"].loc[r["mode"]])
        else:
            out.append(model["glob"])
    return np.asarray(out, float)


# --------------------------------------------------------------------------- #
# Validación temporal forward-chaining
# --------------------------------------------------------------------------- #
def forward_folds(years):
    yrs = sorted(int(y) for y in years if pd.notna(y))
    folds = []
    for i in range(1, len(yrs)):
        tr = [y for y in yrs if y <= yrs[i - 1]]
        va = yrs[i]
        folds.append((tr, va))
    return folds


def fit_member(kind, target_kind, Xtr, dtr, params=None):
    """Entrena UN miembro y devuelve un dict picklable (modelo + corrección de sesgo).

    target_kind:
      - "direct": objetivo log1p(servicios); back-transform con corrección de sesgo.
      - "ratio" : objetivo log1p(servicios/amount_usd); se reconstruye nivel = ratio*amount.
                  amount_usd se conoce a la llegada (no es leakage) y el ratio tiene mucho
                  menos cola que el monto absoluto. Filas con amount<=0 -> fallback baseline.
    """
    if target_kind == "direct":
        ytr_log = np.log1p(np.clip(dtr["target_servicios"].values, 0, None))
        mdl = make_model(kind, params)
        mdl.fit(Xtr, ytr_log)
        corr = log_correction(ytr_log - mdl.predict(Xtr))
        return dict(kind=kind, tkind="direct", model=mdl, corr=corr, base=None)

    amt = pd.to_numeric(dtr["amount_usd"], errors="coerce").values
    ok = amt > 0
    ratio = np.clip(dtr["target_servicios"].values[ok] / amt[ok], 0, None)
    ytr_log = np.log1p(ratio)
    mdl = make_model(kind, params)
    mdl.fit(Xtr[ok], ytr_log)
    corr = log_correction(ytr_log - mdl.predict(Xtr[ok]))
    return dict(kind=kind, tkind="ratio", model=mdl, corr=corr, base=baseline_fit(dtr))


def predict_member(mem, Xva, dva):
    """Predice a NIVEL con un miembro entrenado por fit_member."""
    if mem["tkind"] == "direct":
        return back_transform(mem["model"].predict(Xva), mem["corr"])
    amtv = pd.to_numeric(dva["amount_usd"], errors="coerce").values
    lvl = back_transform(mem["model"].predict(Xva), mem["corr"]) * np.clip(amtv, 0, None)
    bad = ~(amtv > 0)
    if bad.any():
        lvl = lvl.copy()
        lvl[bad] = baseline_predict(mem["base"], dva[bad])
    return lvl


def fit_predict_level(kind, target_kind, Xtr, dtr, params=None):
    """Conveniencia para CV: entrena un miembro y devuelve su predictor a nivel."""
    mem = fit_member(kind, target_kind, Xtr, dtr, params)
    return lambda Xva, dva: predict_member(mem, Xva, dva)


def champion_specs(name):
    """Devuelve la lista de (kind, target_kind) que componen un champion (spec o ensamble)."""
    if name in ENSEMBLES:
        members = ENSEMBLES[name]
    else:
        members = [name]
    by_name = {s[0]: (s[1], s[2]) for s in SPECS}
    return [by_name[m] for m in members]


def fit_champion(name, Xtr, dtr, lgbm_params=None):
    """Entrena todos los miembros del champion. Devuelve lista de dicts picklables."""
    out = []
    for kind, tkind in champion_specs(name):
        p = lgbm_params if kind == "lgbm" else None
        out.append(fit_member(kind, tkind, Xtr, dtr, p))
    return out


def predict_champion(members, Xva, dva):
    """Media de los miembros (1 miembro = modelo simple; >1 = ensamble)."""
    return np.mean([predict_member(m, Xva, dva) for m in members], axis=0)


# specs evaluadas: (nombre, kind, target_kind)
# NOTA: el enfoque "ratio" (servicios/amount_usd) se probó y se DESCARTÓ — los outliers
# de amount_usd amplifican el error y el smearing sobre residuos log-ratio de cola pesada
# dispara la predicción (WAPE>1.3). El camino directo gana con claridad. fit_member("ratio")
# se conserva por si en el futuro se acota amount_usd, pero no entra al pool de campeones.
SPECS = [
    ("lgbm", "lgbm", "direct"),
    ("xgb",  "xgb",  "direct"),
    ("rf",   "rf",   "direct"),
]
# ensambles: media de las predicciones a nivel de los miembros
ENSEMBLES = {
    "ens_direct": ["lgbm", "xgb"],
}


def run_cv(df, lgbm_params=None):
    """Forward-chaining temporal. Evalúa baseline + SPECS (directo/ratio) + ENSEMBLES.
    Devuelve OOF pooled por modelo + métricas por fold."""
    ylevel = df["target_servicios"].values
    Xprep = F.prepare(df)
    anio = pd.to_numeric(df["anio"], errors="coerce")
    folds = forward_folds(anio.dropna().unique())

    names = [s[0] for s in SPECS] + list(ENSEMBLES) + ["baseline"]
    oof = {n: pd.Series(np.nan, index=df.index) for n in names}
    fold_rows = []

    for tr_years, va_year in folds:
        tr = anio.isin(tr_years).values
        va = (anio == va_year).values
        if tr.sum() < 30 or va.sum() == 0:
            continue
        pos = np.where(va)[0]
        Xtr, Xva = Xprep[tr], Xprep[va]
        dtr, dva = df[tr], df[va]

        # baseline
        bm = baseline_fit(dtr)
        pb = baseline_predict(bm, dva)
        oof["baseline"].iloc[pos] = pb
        fold_rows.append(dict(model="baseline", val_year=int(va_year),
                              n_train=int(tr.sum()), **metrics(ylevel[va], pb)))

        # specs
        fold_pred = {}
        for name, kind, tkind in SPECS:
            p = lgbm_params if kind == "lgbm" else None
            predict = fit_predict_level(kind, tkind, Xtr, dtr, p)
            pv = predict(Xva, dva)
            fold_pred[name] = pv
            oof[name].iloc[pos] = pv
            fold_rows.append(dict(model=name, val_year=int(va_year),
                                  n_train=int(tr.sum()), **metrics(ylevel[va], pv)))

        # ensambles
        for ename, members in ENSEMBLES.items():
            pe = np.mean([fold_pred[m] for m in members], axis=0)
            oof[ename].iloc[pos] = pe
            fold_rows.append(dict(model=ename, val_year=int(va_year),
                                  n_train=int(tr.sum()), **metrics(ylevel[va], pe)))

    return oof, pd.DataFrame(fold_rows)


# --------------------------------------------------------------------------- #
# Modelo por concepto
# --------------------------------------------------------------------------- #
def run_cv_per_concept(df, concept_cols):
    ylevel = df["target_servicios"].values
    Xprep = F.prepare(df)
    anio = pd.to_numeric(df["anio"], errors="coerce")
    folds = forward_folds(anio.dropna().unique())
    oof = pd.Series(0.0, index=df.index)
    covered = pd.Series(False, index=df.index)

    for tr_years, va_year in folds:
        tr = anio.isin(tr_years).values
        va = (anio == va_year).values
        if tr.sum() < 30 or va.sum() == 0:
            continue
        Xtr, Xva = Xprep[tr], Xprep[va]
        total_va = np.zeros(int(va.sum()))
        for col in concept_cols:
            ytr_log = np.log1p(np.clip(df[col].values[tr], 0, None))
            mdl = make_model("lgbm")
            mdl.fit(Xtr, ytr_log)
            corr = log_correction(ytr_log - mdl.predict(Xtr))
            total_va += back_transform(mdl.predict(Xva), corr)
        oof.iloc[np.where(va)[0]] = total_va
        covered.iloc[np.where(va)[0]] = True

    mask = covered.values
    return metrics(ylevel[mask], oof.values[mask]), oof, covered


# --------------------------------------------------------------------------- #
# Intervalos cuantílicos
# --------------------------------------------------------------------------- #
def run_cv_quantiles(df, target_col="log_target_servicios", conformal=True):
    """Intervalos P10/P90 con CQR (conformalized quantile regression). Por fold:
    se separa una calibración (30%) del train para recalibrar el margen y alcanzar
    cobertura nominal 80%. Reporta cobertura cruda y conformalizada."""
    ylevel = df["target_servicios"].values
    Xprep = F.prepare(df)
    anio = pd.to_numeric(df["anio"], errors="coerce")
    folds = forward_folds(anio.dropna().unique())
    p10 = pd.Series(np.nan, index=df.index)   # crudo
    p90 = pd.Series(np.nan, index=df.index)
    c10 = pd.Series(np.nan, index=df.index)   # conformal
    c90 = pd.Series(np.nan, index=df.index)
    rng = np.random.RandomState(SEED)

    for tr_years, va_year in folds:
        tr = anio.isin(tr_years).values
        va = (anio == va_year).values
        if tr.sum() < 40 or va.sum() == 0:
            continue
        Xtr, Xva = Xprep[tr], Xprep[va]
        ytr_log = df[target_col].values[tr]
        n = len(ytr_log); idx = rng.permutation(n)
        ncal = max(20, int(0.30 * n)); cal, fit = idx[:ncal], idx[ncal:]
        q10 = make_quantile(0.1); q10.fit(Xtr.iloc[fit], ytr_log[fit])
        q90 = make_quantile(0.9); q90.fit(Xtr.iloc[fit], ytr_log[fit])
        margin = conformal_margin(q10, q90, Xtr.iloc[cal], ytr_log[cal]) if conformal else 0.0
        lo_log, hi_log = q10.predict(Xva), q90.predict(Xva)
        pos = np.where(va)[0]
        p10.iloc[pos] = np.clip(np.expm1(lo_log), 0, None)
        p90.iloc[pos] = np.clip(np.expm1(hi_log), 0, None)
        c10.iloc[pos] = np.clip(np.expm1(lo_log - margin), 0, None)
        c90.iloc[pos] = np.clip(np.expm1(hi_log + margin), 0, None)

    def _cov(lo_s, hi_s):
        m = lo_s.notna().values
        lo, hi, y = lo_s.values[m], np.maximum(hi_s.values[m], lo_s.values[m]), ylevel[m]
        return float(np.mean((y >= lo) & (y <= hi))), float(np.median(hi - lo)), int(m.sum())

    cov_r, w_r, n_r = _cov(p10, p90)
    cov_c, w_c, _ = _cov(c10, c90)
    return dict(coverage_raw=cov_r, width_raw=w_r,
                coverage_conformal=cov_c, width_conformal=w_c,
                target_nominal=0.80, n=n_r), c10, c90


# --------------------------------------------------------------------------- #
# Tuning (Optuna) del LightGBM — Fase 1.3
# --------------------------------------------------------------------------- #
def _cv_score(df, lgbm_params, target_kind="direct"):
    """WAPE medio del LightGBM sobre folds con historia suficiente (excluye el fold
    2021 sin historia). Métrica robusta para Optuna; valida en forward-chaining."""
    ylevel = df["target_servicios"].values
    Xprep = F.prepare(df)
    anio = pd.to_numeric(df["anio"], errors="coerce")
    wapes = []
    for tr_years, va_year in forward_folds(anio.dropna().unique()):
        tr = anio.isin(tr_years).values
        va = (anio == va_year).values
        if tr.sum() < 60 or va.sum() < 20:          # salta folds data-starved
            continue
        predict = fit_predict_level("lgbm", target_kind, Xprep[tr], df[tr], lgbm_params)
        pv = predict(Xprep[va], df[va])
        wapes.append(metrics(ylevel[va], pv)["WAPE"])
    return float(np.mean(wapes)) if wapes else 1.0


def tune_lgbm(df, target_kind="direct", n_trials=40):
    """Búsqueda de hiperparámetros con Optuna optimizando WAPE en CV temporal."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1200, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 60),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )
        return _cv_score(df, params, target_kind)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# --------------------------------------------------------------------------- #
# Validación robusta — Fase 2
# --------------------------------------------------------------------------- #
def forward_folds_period(periods):
    """Forward-chaining sobre periodos ordenables (p.ej. año*4+trimestre)."""
    ps = sorted(set(int(p) for p in periods if pd.notna(p)))
    return [([p for p in ps if p <= ps[i - 1]], ps[i]) for i in range(1, len(ps))]


def cv_quarterly(df, name, kind, target_kind, lgbm_params=None):
    """OOF de UN modelo bajo forward-chaining TRIMESTRAL (más folds = WAPE/MdAPE
    menos ruidoso). Devuelve métricas pooled sobre filas validadas."""
    ylevel = df["target_servicios"].values
    Xprep = F.prepare(df)
    anio = pd.to_numeric(df["anio"], errors="coerce").to_numpy(dtype=float)
    tri = pd.to_numeric(df.get("trimestre_arribo"), errors="coerce").to_numpy(dtype=float)
    period = anio * 4 + (tri - 1)                       # float ndarray; NA -> nan
    oof = pd.Series(np.nan, index=df.index)
    for tr_periods, va_p in forward_folds_period(period[~np.isnan(period)]):
        tr = np.isin(period, tr_periods)
        va = period == va_p
        if tr.sum() < 60 or va.sum() < 8:
            continue
        p = lgbm_params if kind == "lgbm" else None
        predict = fit_predict_level(kind, target_kind, Xprep[tr], df[tr], p)
        oof.iloc[np.where(va)[0]] = predict(Xprep[va], df[va])
    m = oof.notna().values
    return metrics(ylevel[m], oof.values[m])


def deploy_stability(df, kind, target_kind, deploy_year, lgbm_params=None,
                     seeds=(42, 1, 7, 123, 2024)):
    """Re-evalúa el fold de despliegue variando la semilla del LightGBM; reporta
    media ± desviación de WAPE/MdAPE (Fase 2: el número deja de ser un punto frágil)."""
    Xprep = F.prepare(df)
    anio = pd.to_numeric(df["anio"], errors="coerce").to_numpy(dtype=float)
    years = np.unique(anio[~np.isnan(anio)])
    tr = np.isin(anio, [y for y in years if y < deploy_year])
    va = anio == deploy_year
    yv = df["target_servicios"].values[va]
    waps, mdapes = [], []
    for s in seeds:
        p = dict(lgbm_params or {}); p["random_state"] = int(s)
        predict = fit_predict_level(kind, target_kind, Xprep[tr], df[tr],
                                    p if kind == "lgbm" else None)
        mm = metrics(yv, predict(Xprep[va], df[va]))
        waps.append(mm["WAPE"]); mdapes.append(mm["MdAPE"])
    return dict(WAPE_mean=float(np.mean(waps)), WAPE_std=float(np.std(waps)),
                MdAPE_mean=float(np.mean(mdapes)), MdAPE_std=float(np.std(mdapes)),
                n_seeds=len(seeds))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
N_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "40"))   # tuning de Optuna (Fase 1.3)
ALL_NAMES = [s[0] for s in SPECS] + list(ENSEMBLES) + ["baseline"]


def _overall(oof, ylevel, names):
    out = {}
    for k in names:
        m = oof[k].notna().values
        if m.any():
            out[k] = metrics(ylevel[m], oof[k].values[m])
    return out


def _deploy_table(fold_df, deploy_year):
    return (fold_df[fold_df["val_year"] == deploy_year]
            .set_index("model")[["n", "MAE", "RMSE", "WAPE", "MdAPE", "bias"]])


def main():
    print("Cargando datos...")
    full = D.load(fiable_only=False)
    df = full[full["target_fiable"].astype(bool)].copy().reset_index(drop=True)
    pend = full[~full["target_fiable"].astype(bool)].copy().reset_index(drop=True)
    print(f"fiable={len(df)}  pendientes={len(pend)}")
    ylevel = df["target_servicios"].values

    # ---- 1. CV base (params por defecto): baseline + directo + ratio + ensambles ----
    print("\n[1] Forward-chaining base (default params): directo vs ratio vs ensamble...")
    oof0, fold0 = run_cv(df)
    deploy_year = int(fold0[fold0["n"] >= 20]["val_year"].max())
    dep0 = _deploy_table(fold0, deploy_year)
    ml_names = [s[0] for s in SPECS] + list(ENSEMBLES)
    champ0 = dep0.loc[[n for n in ml_names if n in dep0.index], "WAPE"].idxmin()
    tune_kind = "ratio" if any(tk == "ratio" for _, tk in champion_specs(champ0)) else "direct"
    print(dep0.round(3))
    print(f"-> mejor enfoque (deploy {deploy_year}, default): {champ0}  | tuning target='{tune_kind}'")

    # ---- 1b. Tuning Optuna del LightGBM sobre el enfoque ganador ----
    print(f"\n[1b] Tuning Optuna ({N_TRIALS} trials, objetivo WAPE forward-chaining)...")
    best_params, best_val = tune_lgbm(df, tune_kind, n_trials=N_TRIALS)
    print(f"mejor WAPE-CV={best_val:.4f}  params={best_params}")

    # ---- 1c. Re-CV con LightGBM tuneado; elegir champion final ----
    print("\n[1c] Re-evaluando con LightGBM tuneado...")
    oof, fold_df = run_cv(df, lgbm_params=best_params)
    fold_df.to_csv(os.path.join(C.REPORTS_ML, "metrics_por_fold.csv"), index=False)
    overall = _overall(oof, ylevel, ALL_NAMES)
    pd.DataFrame(overall).T.to_csv(os.path.join(C.REPORTS_ML, "metrics_global.csv"))
    deploy = _deploy_table(fold_df, deploy_year)
    champion = deploy.loc[[n for n in ml_names if n in deploy.index], "WAPE"].idxmin()
    print("Pooled (todas las ventanas):"); print(pd.DataFrame(overall).T.round(3))
    print(f"\nFold de despliegue (val {deploy_year}):"); print(deploy.round(3))
    print(f"-> CHAMPION = {champion}  (WAPE deploy {deploy.loc[champion,'WAPE']:.3f}, "
          f"MdAPE {deploy.loc[champion,'MdAPE']:.3f})")

    # ---- 2. Métricas por segmento (champion, OOF pooled) ----
    print("\n[2] Métricas por segmento (champion OOF)...")
    eval_mask = oof[champion].notna().values
    seg_df = df.loc[eval_mask, SEG].copy()
    seg_df["y"] = ylevel[eval_mask]
    seg_df["yhat"] = oof[champion].values[eval_mask]
    rows = []
    for key, g in seg_df.groupby(SEG, observed=True):
        rows.append(dict(mode=key[0], categoria_canon=key[1],
                         **metrics(g["y"].values, g["yhat"].values)))
    seg_metrics = pd.DataFrame(rows).sort_values("n", ascending=False)
    seg_metrics.to_csv(os.path.join(C.REPORTS_ML, "metrics_por_segmento.csv"), index=False)
    print(seg_metrics.round(2).to_string(index=False))

    # ---- 3. Modelo por concepto ----
    print("\n[3] Modelo por concepto (suma de conceptos)...")
    tc = pd.read_parquet(os.path.join(C.PROC_DIR, "target_por_concepto.parquet"))
    concept_cols = [c for c in tc.columns if c not in ("op_id_full", "DERECHOS_IMPUESTOS")]
    dfc = df.merge(tc, on="op_id_full", how="left")
    for c in concept_cols:
        dfc[c] = pd.to_numeric(dfc[c], errors="coerce").fillna(0.0)
    pc_metrics, _, _ = run_cv_per_concept(dfc, concept_cols)
    print("por-concepto:", {k: round(v, 3) for k, v in pc_metrics.items()})

    # ---- 4. Intervalos ----
    print("\n[4] Intervalos cuantílicos P10/P90...")
    qinfo, _, _ = run_cv_quantiles(df)
    print("intervalos:", {k: round(v, 3) if isinstance(v, float) else v
                          for k, v in qinfo.items()})

    # ---- 4b. Validación robusta (Fase 2): folds trimestrales + multi-semilla ----
    print("\n[4b] Validación robusta (Fase 2)...")
    quarterly = cv_quarterly(df, "lgbm", "lgbm", tune_kind, best_params)
    stability = deploy_stability(df, "lgbm", tune_kind, deploy_year, best_params)
    print("trimestral (lgbm):", {k: round(v, 3) for k, v in quarterly.items()})
    print("estabilidad deploy (multi-seed):", {k: round(v, 4) for k, v in stability.items()})

    # ---- 5. Modelo final (champion sobre todo el fiable) + importancias ----
    print(f"\n[5] Entrenando champion final ({champion}) sobre todo el fiable...")
    Xall = F.prepare(df)
    ylog_all = df["log_target_servicios"].values
    members = fit_champion(champion, Xall, df, best_params)     # picklable

    # cuantiles + margen conformal (calib 30%), luego refit en todo el fiable
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(ylog_all)); ncal = int(0.30 * len(ylog_all))
    cal, fit = idx[:ncal], idx[ncal:]
    q10c = make_quantile(0.1); q10c.fit(Xall.iloc[fit], ylog_all[fit])
    q90c = make_quantile(0.9); q90c.fit(Xall.iloc[fit], ylog_all[fit])
    conf_margin = conformal_margin(q10c, q90c, Xall.iloc[cal], ylog_all[cal])
    q10f = make_quantile(0.1); q10f.fit(Xall, ylog_all)
    q90f = make_quantile(0.9); q90f.fit(Xall, ylog_all)

    # importancias (modelo lgbm directo tuneado, para interpretabilidad)
    imp_model = make_model("lgbm", best_params); imp_model.fit(Xall, ylog_all)
    ct = imp_model.named_steps["ct"]
    imp = pd.DataFrame({"feature": list(ct.get_feature_names_out()),
                        "gain": imp_model.named_steps["m"].feature_importances_})
    imp = imp.sort_values("gain", ascending=False).head(25)
    imp.to_csv(os.path.join(C.REPORTS_ML, "feature_importance.csv"), index=False)
    print(imp.head(15).to_string(index=False))

    # ---- 6. Predicción de pendientes (champion) ----
    print("\n[6] Prediciendo operaciones pendientes...")
    pred_pend = pd.DataFrame()
    if len(pend):
        Xp = F.prepare(pend)
        yhat = predict_champion(members, Xp, pend)
        lo = np.clip(np.expm1(q10f.predict(Xp) - conf_margin), 0, None)
        hi = np.maximum(np.clip(np.expm1(q90f.predict(Xp) + conf_margin), 0, None), lo)
        pred_pend = pd.DataFrame({
            "op_id_full": pend["op_id_full"].values,
            "anio": pd.to_numeric(pend["anio"], errors="coerce").values,
            "mode": pend["mode"].values,
            "categoria_canon": pend["categoria_canon"].values,
            "status": pend["status"].values,
            "servicios_pred_usd": np.round(yhat, 1),
            "p10_usd": np.round(lo, 1),
            "p90_usd": np.round(hi, 1),
        })
        pred_pend.to_csv(os.path.join(C.REPORTS_ML, "predicciones_pendientes.csv"),
                         index=False)
        print(f"guardado: predicciones_pendientes.csv ({len(pred_pend)} ops)")
        print("suma servicios prevista (pendientes): USD",
              round(float(pred_pend["servicios_pred_usd"].sum()), 0))

    # ---- 7. Guardar modelos ----
    joblib.dump(dict(champion=champion, members=members, lgbm_params=best_params,
                     corr_method=CORR_METHOD),
                os.path.join(MODELS_DIR, "champion_servicios.pkl"))
    # compat: modelo lgbm directo (interpretabilidad / consumidores externos)
    joblib.dump(imp_model, os.path.join(MODELS_DIR, "lgbm_servicios.pkl"))
    joblib.dump(q10f, os.path.join(MODELS_DIR, "lgbm_servicios_p10.pkl"))
    joblib.dump(q90f, os.path.join(MODELS_DIR, "lgbm_servicios_p90.pkl"))
    joblib.dump(baseline_fit(df), os.path.join(MODELS_DIR, "baseline_segmento.pkl"))
    meta = dict(target="target_servicios (log1p)", champion=champion,
                target_kind=tune_kind, corr_method=CORR_METHOD,
                lgbm_params=best_params, conformal_margin_log=conf_margin,
                features_num_log=F.NUM_LOG, features_num=F.NUM_PLAIN + F.DERIVED + F.FLAGS,
                features_oh=F.CAT_OH, features_te=F.CAT_TE,
                n_train=len(df), seed=SEED, version="2-fase1-2")
    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=float)

    # ---- 8. Persistir resúmenes para el reporte ----
    summary = dict(overall=overall,
                   deploy_year=deploy_year,
                   deploy=deploy.to_dict(orient="index"),
                   deploy_default=dep0.to_dict(orient="index"),
                   champion=champion, champion_default=champ0, tune_kind=tune_kind,
                   tuning=dict(best_params=best_params, best_wape_cv=best_val,
                               n_trials=N_TRIALS),
                   quarterly=quarterly, stability=stability,
                   per_concept=pc_metrics, quantiles=qinfo,
                   conformal_margin_log=conf_margin,
                   pending_total=float(pred_pend["servicios_pred_usd"].sum()) if len(pred_pend) else 0.0,
                   pending_n=int(len(pred_pend)))
    with open(os.path.join(C.REPORTS_ML, "_modelado_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nOK. Champion={champion}. Artefactos en models/ y reports/.")


if __name__ == "__main__":
    main()
