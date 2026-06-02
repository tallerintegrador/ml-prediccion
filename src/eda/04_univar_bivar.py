# -*- coding: utf-8 -*-
"""
04_univar_bivar.py — Análisis univariado (4.3) y bivariado (4.4).

Sobre el subconjunto de TARGET FIABLE. Target primario = servicios logísticos.
  4.3  distribución del target (servicios y tributos), numéricas, categóricas (cardinalidad)
  4.4  costo vs modo/incoterm/categoría/país/agente/canal (Kruskal-Wallis + η²),
       costo vs escala (peso/valor/qty: Pearson/Spearman), costo unitario, correlaciones
Genera reports/06_univariado.md, reports/07_bivariado.md y figuras f04_*, f05_*.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import config as C
import utils as U
import dataset as D

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
TGT = "target_servicios"


def kruskal_eta2(df, group, value, min_n=15):
    """Kruskal-Wallis + epsilon² (tamaño del efecto). Filtra grupos con n<min_n."""
    sub = df[[group, value]].dropna()
    counts = sub[group].value_counts()
    keep = counts[counts >= min_n].index
    sub = sub[sub[group].isin(keep)]
    groups = [g[value].values for _, g in sub.groupby(group)]
    if len(groups) < 2:
        return np.nan, np.nan, 0
    H, p = stats.kruskal(*groups)
    n = len(sub); k = len(groups)
    eps2 = (H - k + 1) / (n - k) if n > k else np.nan   # epsilon-cuadrado
    return p, eps2, len(keep)


def main():
    df = D.load(fiable_only=True)
    n = len(df)
    R = U.MdReport(f"4.3 — Análisis univariado (target fiable, n={n})")

    # ---------------- 4.3 target ----------------
    R.h("Distribución del target — servicios logísticos (USD)")
    for col, lab in [("target_servicios", "servicios"), ("tributos_usd", "tributos"),
                     ("target_total", "total")]:
        s = df[col].dropna()
        R.p(f"- **{lab}**: media={s.mean():.0f}, mediana={s.median():.0f}, "
            f"p5={s.quantile(.05):.0f}, p95={s.quantile(.95):.0f}, "
            f"skew={s.skew():.2f}, curtosis={s.kurtosis():.2f}, "
            f"CV={s.std()/s.mean():.2f}")
    s = df[TGT].dropna(); s = s[s > 0]
    R.p(f"\nLog-normalidad (servicios): Shapiro p={stats.shapiro(s.sample(min(500,len(s)),random_state=C.SEED))[1]:.1e} "
        f"crudo vs p={stats.shapiro(np.log1p(s.sample(min(500,len(s)),random_state=C.SEED)))[1]:.1e} en log.")
    # outliers IQR
    q1, q3 = s.quantile(.25), s.quantile(.75); iqr = q3 - q1
    hi = s[s > q3 + 3 * iqr]
    R.p(f"- Outliers severos (>Q3+3·IQR = >{q3+3*iqr:.0f}): {len(hi)} ops "
        f"({100*len(hi)/len(s):.1f}%), máx={s.max():.0f}. No se eliminan (operaciones caras legítimas).")

    # fig target servicios vs tributos
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(np.log1p(df[TGT].dropna()), bins=50, color="#2b7bba")
    ax[0].set_title("log(1+servicios)")
    ax[1].hist(np.log1p(df["tributos_usd"][df["tributos_usd"] > 0]), bins=50, color="#c44e52")
    ax[1].set_title("log(1+tributos)")
    ax[2].scatter(np.log1p(df["tributos_usd"]), np.log1p(df[TGT]), s=8, alpha=.4)
    ax[2].set_xlabel("log tributos"); ax[2].set_ylabel("log servicios")
    ax[2].set_title(f"servicios vs tributos (ρ={df[['tributos_usd',TGT]].corr('spearman').iloc[0,1]:.2f})")
    fig.suptitle("4.3 — Target: servicios vs tributos", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_EDA, "f04_target.png")); plt.close(fig)

    # ---------------- numéricas ----------------
    R.h("Variables numéricas")
    nums = ["amount_usd", "qty", "bulks", "ctnr_qty", "peso_bruto", "payment_term",
            "transit_days", "n_oc_distintas", "n_productos"]
    desc = df[nums].describe(percentiles=[.05, .5, .95]).T[["count", "mean", "50%", "95%", "max"]]
    desc["%nulos"] = [round(100 * df[c].isna().mean(), 1) for c in nums]
    R.p(desc.round(1).to_markdown())

    # ---------------- categóricas ----------------
    R.h("Variables categóricas — cardinalidad y categorías raras")
    cats = D.FEATURES_CAT
    rows = []
    for c in cats:
        if c not in df:
            continue
        vc = df[c].value_counts()
        raras = (vc < 15).sum()
        rows.append(dict(variable=c, cardinalidad=df[c].nunique(),
                         pct_nulos=round(100 * df[c].isna().mean(), 1),
                         top=f"{vc.index[0]} ({vc.iloc[0]})" if len(vc) else "—",
                         cat_raras_n=int(raras)))
    cardf = pd.DataFrame(rows).sort_values("cardinalidad", ascending=False)
    R.p(cardf.to_markdown(index=False))
    R.p("\n- **Alta cardinalidad** (supplier, pol, producto): requieren target/frequency "
        "encoding, no one-hot.\n- **Categorías raras** (n<15): agrupar en 'OTROS' o "
        "usar suavizado bayesiano para evitar sobreajuste.")
    R.save(os.path.join(C.REPORTS_EDA, "06_univariado.md"))

    # ================= 4.4 BIVARIADO =================
    B = U.MdReport(f"4.4 — Análisis bivariado: drivers del costo de servicios (n={n})")

    # --- categóricas vs target: Kruskal + eta2 ---
    B.h("Costo de servicios por variable categórica (Kruskal-Wallis + ε²)")
    drivers = ["mode", "type", "incoterm", "incoterm_grupo", "categoria_canon",
               "pais_origen", "customs_agent", "ffw", "shipping_line", "depot", "canal"]
    rows = []
    for g in drivers:
        if g not in df:
            continue
        p, eps2, k = kruskal_eta2(df, g, TGT)
        rows.append(dict(variable=g, n_grupos=k, kruskal_p=p, epsilon2=eps2))
    eff = pd.DataFrame(rows).dropna(subset=["epsilon2"]).sort_values("epsilon2", ascending=False)
    eff["efecto"] = pd.cut(eff["epsilon2"], [-1, .01, .06, .14, 1],
                           labels=["insignif.", "pequeño", "mediano", "grande"])
    B.p(eff.assign(kruskal_p=eff["kruskal_p"].map(lambda x: f"{x:.1e}"),
                   epsilon2=eff["epsilon2"].round(3)).to_markdown(index=False))
    B.p("\nε² (epsilon-cuadrado): >0.14 efecto grande, 0.06–0.14 mediano, <0.06 pequeño.")

    # fig boxplots de los 4 drivers categóricos top
    top_drivers = eff["variable"].head(4).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, g in zip(axes.ravel(), top_drivers):
        order = df.groupby(g)[TGT].median().sort_values().index
        d = df[df[g].isin(order)]
        sns.boxplot(data=d, x=g, y=TGT, order=order, ax=ax, showfliers=False)
        ax.set_yscale("log"); ax.set_title(f"{g} (ε²={eff.set_index('variable').loc[g,'epsilon2']:.2f})")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.suptitle("4.4 — Costo de servicios por driver categórico (log)", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_EDA, "f05_drivers_cat.png")); plt.close(fig)

    # --- modo aéreo vs marítimo: costo unitario ---
    B.h("Hipótesis: aéreo más caro por kg")
    for modo in ["AIR", "SEA"]:
        d = df[df["mode"] == modo]
        B.p(f"- **{modo}** (n={len(d)}): servicios mediana=${d[TGT].median():.0f}; "
            f"costo/valor mediano={d['costo_unit_valor'].median():.3f} "
            f"(servicios por USD de mercadería)")

    # --- numéricas de escala vs target ---
    B.h("Costo vs variables de escala (Pearson/Spearman)")
    rows = []
    for v in ["amount_usd", "qty", "bulks", "ctnr_qty", "peso_bruto", "transit_days"]:
        d = df[[v, TGT]].dropna()
        d = d[(d[v] > 0)]
        if len(d) < 20:
            continue
        pr = stats.pearsonr(np.log1p(d[v]), np.log1p(d[TGT]))[0]
        sp = stats.spearmanr(d[v], d[TGT])[0]
        rows.append(dict(variable=v, n=len(d), pearson_log=round(pr, 3),
                         spearman=round(sp, 3)))
    B.p(pd.DataFrame(rows).sort_values("spearman", ascending=False, key=abs).to_markdown(index=False))

    # fig scatter escala
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, v in zip(axes, ["amount_usd", "peso_bruto", "ctnr_qty"]):
        d = df[[v, TGT, "mode"]].dropna(); d = d[d[v] > 0]
        for modo, col in [("AIR", "#e0a458"), ("SEA", "#2b7bba")]:
            dd = d[d["mode"] == modo]
            ax.scatter(dd[v], dd[TGT], s=10, alpha=.4, color=col, label=modo)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(v); ax.set_ylabel("servicios USD"); ax.legend(fontsize=7)
    fig.suptitle("4.4 — Costo de servicios vs escala (log-log, color=modo)", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_EDA, "f05_escala.png")); plt.close(fig)

    # --- matriz de correlación ---
    B.h("Matriz de correlación (Spearman) entre numéricas y target")
    cm = ["amount_usd", "qty", "bulks", "ctnr_qty", "peso_bruto", "payment_term",
          "transit_days", TGT, "tributos_usd"]
    corr = df[cm].corr("spearman")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Correlación de Spearman")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_EDA, "f05_corr.png")); plt.close(fig)
    # multicolinealidad
    hi = [(cm[i], cm[j], corr.iloc[i, j]) for i in range(len(cm)) for j in range(i+1, len(cm))
          if abs(corr.iloc[i, j]) > 0.6 and TGT not in (cm[i],)]
    B.p("\n**Multicolinealidad (|ρ|>0.6 entre predictores):** " +
        ("; ".join(f"{a}~{b}={r:.2f}" for a, b, r in hi) if hi else "ninguna fuerte."))

    B.save(os.path.join(C.REPORTS_EDA, "07_bivariado.md"))
    print("OK 4.3/4.4 ->", "06_univariado.md, 07_bivariado.md")
    print("drivers top:", eff[["variable", "epsilon2"]].head(5).to_dict("records"))


if __name__ == "__main__":
    main()
