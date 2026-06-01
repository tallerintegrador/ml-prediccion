# -*- coding: utf-8 -*-
"""09_figuras.py — Figuras del modelado a partir de los CSV/JSON de reports/."""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import config as C

R = C.REPORTS_ML
FIG = C.FIG_ML
summary = json.load(open(os.path.join(R, "_modelado_summary.json"), encoding="utf-8"))

# orden de modelos a graficar: base + champion (si es ratio/ensamble)
_base = ["baseline", "rf", "lgbm", "xgb"]
_champ = summary.get("champion", "lgbm")
ORDER = _base + ([_champ] if _champ not in _base else [])

# 1) Comparación de modelos (fold de despliegue) -------------------------------
dep = pd.DataFrame(summary["deploy"]).T.reindex(ORDER)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
dep["WAPE"].plot.bar(ax=ax[0], color="#4C72B0"); ax[0].set_title(f"WAPE — fold despliegue (val {summary['deploy_year']})")
dep["MdAPE"].plot.bar(ax=ax[1], color="#55A868"); ax[1].set_title("MdAPE — fold despliegue")
for a in ax:
    a.set_xlabel(""); a.tick_params(axis="x", rotation=0)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "14_modelos_despliegue.png"), dpi=110)
plt.close(fig)

# 2) WAPE por segmento (n>=10) -------------------------------------------------
seg = pd.read_csv(os.path.join(R, "metrics_por_segmento.csv"))
seg = seg[seg["n"] >= 10].copy()
seg["seg"] = seg["mode"] + " × " + seg["categoria_canon"]
seg = seg.sort_values("WAPE")
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(seg["seg"], seg["WAPE"], color="#C44E52")
ax.set_xlabel("WAPE (LGBM, OOF forward-chaining)")
ax.set_title("Error por segmento mode × categoria_canon (n≥10)")
for i, (w, n) in enumerate(zip(seg["WAPE"], seg["n"])):
    ax.text(w + 0.01, i, f"n={n}", va="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "14_wape_por_segmento.png"), dpi=110)
plt.close(fig)

# 3) Pooled vs despliegue (MdAPE) ----------------------------------------------
ov = pd.DataFrame(summary["overall"]).T["MdAPE"]
dp = dep["MdAPE"]
comp = pd.DataFrame({"pooled (todas las ventanas)": ov, f"despliegue (val {summary['deploy_year']})": dp})
fig, ax = plt.subplots(figsize=(7, 4))
comp.reindex(ORDER).plot.bar(ax=ax)
ax.set_title("MdAPE: ventana pooled vs fold de despliegue"); ax.set_xlabel("")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "14_mdape_pooled_vs_deploy.png"), dpi=110)
plt.close(fig)

print("Figuras guardadas en", FIG)
