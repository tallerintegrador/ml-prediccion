# -*- coding: utf-8 -*-
"""
backtest_registradas.py — PRUEBA SOBRE DATOS YA REGISTRADOS.

Toma las operaciones que YA tienen costo real (target fiable), predice su costo de
servicios y lo compara con lo que realmente costó -> mide "cuánto falta" para el 90%
(error <=10%).

Dos modos:
  --modo insample  (default, rápido): usa el champion ya guardado (models/) para predecir
                    TODAS las operaciones fiables. OJO: optimista, el modelo vio estos datos.
  --modo honesto   : validación temporal forward-chaining (entrena con años < corte y
                    predice el año de corte, que el modelo NUNCA vio). Es la cifra real
                    de despliegue (~25% MdAPE). Reentrena -> tarda más.

Salida: métricas globales + por año + peores casos + CSV en reports/modelado/.

Uso:
  python src/ml/backtest_registradas.py                 # in-sample, todos los años
  python src/ml/backtest_registradas.py --modo honesto  # holdout temporal del último año
  python src/ml/backtest_registradas.py --modo honesto --corte 2024
"""
import os
import sys
import argparse
import importlib.util
import numpy as np
import pandas as pd

import _predecir_lib as P
import dataset as D          # ya en sys.path vía _predecir_lib
import config as C
import features as F

OBJ_MDAPE = 0.10             # meta "90%": error mediano <= 10%
APE_FLOOR = 50.0             # ignora % sobre operaciones casi-cero (mismo criterio que 08_modelado)


def metrics(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    err = yhat - y
    mask = y >= APE_FLOOR
    mdape = float(np.median(np.abs(err[mask] / y[mask]))) if mask.sum() else np.nan
    return dict(n=int(len(y)),
                MAE=float(np.mean(np.abs(err))),
                RMSE=float(np.sqrt(np.mean(err ** 2))),
                WAPE=float(np.sum(np.abs(err)) / max(np.sum(np.abs(y)), 1e-9)),
                MdAPE=mdape,
                bias=float(np.mean(err)))


def _cargar_modulo_modelado():
    """Importa src/ml/08_modelado.py (nombre empieza por dígito -> via importlib)."""
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "08_modelado.py")
    spec = importlib.util.spec_from_file_location("modelado_mod", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def predecir_insample(df):
    """Predice todas las filas con el champion guardado (rápido, optimista)."""
    art = P.load_artifacts()
    res = P.predecir(art, df)
    return res["servicios_pred_usd"].values


def predecir_honesto(df, corte=None):
    """Forward-chaining: entrena con años < corte, predice el año 'corte' (no visto)."""
    M = _cargar_modulo_modelado()
    art = P.load_artifacts()
    champion = art["champion"]
    lgbm_params = art["meta"]["lgbm_params"]

    anio = pd.to_numeric(df["anio"], errors="coerce")
    años = sorted(int(a) for a in anio.dropna().unique())
    if corte is None:
        # último año con suficientes operaciones (>=20), igual que el deploy_year del entrenamiento
        con_datos = [a for a in años if (anio == a).sum() >= 20]
        corte = con_datos[-1] if con_datos else años[-1]
    tr = (anio < corte).values
    va = (anio == corte).values
    if tr.sum() < 30 or va.sum() == 0:
        raise SystemExit(f"Año de corte {corte} sin datos suficientes (train={tr.sum()}, val={va.sum()}).")

    Xprep = F.prepare(df)
    members = M.fit_champion(champion, Xprep[tr], df[tr], lgbm_params)
    yhat_va = M.predict_champion(members, Xprep[va], df[va])

    yhat = np.full(len(df), np.nan)
    yhat[np.where(va)[0]] = yhat_va
    return yhat, corte


def main():
    ap = argparse.ArgumentParser(description="Backtest del modelo sobre operaciones ya registradas.")
    ap.add_argument("--modo", choices=["insample", "honesto"], default="insample")
    ap.add_argument("--corte", type=int, default=None, help="año a predecir en modo honesto")
    ap.add_argument("--top", type=int, default=15, help="cuántos peores casos mostrar")
    args = ap.parse_args()

    print("Cargando operaciones fiables...")
    df = D.load(fiable_only=True).reset_index(drop=True)
    y = df["target_servicios"].values
    print(f"  {len(df)} operaciones con costo real.")

    if args.modo == "insample":
        print("\nMODO IN-SAMPLE (champion guardado, optimista — el modelo vio estos datos).")
        yhat = predecir_insample(df)
        eval_mask = np.ones(len(df), bool)
        corte = None
    else:
        print("\nMODO HONESTO (forward-chaining: entrena con el pasado, predice año no visto).")
        yhat, corte = predecir_honesto(df, args.corte)
        eval_mask = ~np.isnan(yhat)
        print(f"  Año de corte (holdout) = {corte}: {eval_mask.sum()} operaciones evaluadas.")

    ym, yhm = y[eval_mask], yhat[eval_mask]

    # --- tabla de comparación operación por operación ---
    comp = df.loc[eval_mask, ["op_id_full", "anio", "mode", "categoria_canon",
                              "amount_usd"]].copy()
    comp["real_usd"] = np.round(ym, 1)
    comp["pred_usd"] = np.round(yhm, 1)
    comp["error_usd"] = np.round(yhm - ym, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        comp["error_pct"] = np.where(ym >= APE_FLOOR,
                                     np.round(100 * np.abs(yhm - ym) / ym, 1), np.nan)
    comp = comp.sort_values("error_pct", ascending=False, na_position="last")

    # --- métricas globales ---
    glob = metrics(ym, yhm)
    print("\n" + "=" * 60)
    print(" MÉTRICAS GLOBALES (predicho vs real)")
    print("=" * 60)
    print(f"  operaciones evaluadas : {glob['n']}")
    print(f"  MAE   (error medio $) : USD {glob['MAE']:,.0f}")
    print(f"  RMSE                  : USD {glob['RMSE']:,.0f}")
    print(f"  WAPE  (error agregado): {glob['WAPE']:.3f}   (meta <= 0.10)")
    print(f"  MdAPE (error mediano) : {glob['MdAPE']:.3f}  = {glob['MdAPE']*100:.1f}%   (meta <= 10%)")
    print(f"  bias  (sesgo medio $) : USD {glob['bias']:,.0f}")

    # --- ¿cuánto falta para el 90%? ---
    print("\n" + "=" * 60)
    print(" ¿CUÁNTO FALTA PARA EL 90% (error <= 10%)?")
    print("=" * 60)
    falta_mdape = max(glob["MdAPE"] - OBJ_MDAPE, 0)
    falta_wape = max(glob["WAPE"] - 0.10, 0)
    print(f"  MdAPE actual {glob['MdAPE']*100:.1f}%  ->  meta 10%   | falta recortar {falta_mdape*100:.1f} pts")
    print(f"  WAPE  actual {glob['WAPE']:.3f} ->  meta 0.10  | falta recortar {falta_wape:.3f}")
    dentro = (comp["error_pct"] <= OBJ_MDAPE * 100).mean()
    print(f"  Operaciones ya dentro de ±10%: {dentro*100:.1f}%")
    print("  (El salto a 10% NO es algoritmo: faltan datos — HS, flete cotizado,")
    print("   peso, tarifarios. Ver reports/modelado/16_requisitos_90pct.md)")

    # --- por año ---
    print("\n" + "-" * 60)
    print(" ERROR POR AÑO DE ARRIBO")
    print("-" * 60)
    filas = []
    for a, g in comp.groupby("anio"):
        mm = metrics(g["real_usd"].values, g["pred_usd"].values)
        filas.append(dict(anio=int(a) if pd.notna(a) else -1, **mm))
    por_anio = pd.DataFrame(filas).sort_values("anio")
    print(por_anio[["anio", "n", "WAPE", "MdAPE", "bias"]].round(3).to_string(index=False))

    # --- peores casos ---
    print("\n" + "-" * 60)
    print(f" TOP {args.top} PEORES PREDICCIONES (mayor error %)")
    print("-" * 60)
    cols = ["op_id_full", "anio", "mode", "categoria_canon", "real_usd", "pred_usd", "error_pct"]
    print(comp[cols].head(args.top).to_string(index=False))

    # --- guardar ---
    sufijo = "insample" if args.modo == "insample" else f"honesto_{corte}"
    out = os.path.join(C.REPORTS_ML, f"backtest_{sufijo}.csv")
    comp.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nGuardado detalle operación-por-operación: {out}")


if __name__ == "__main__":
    main()
