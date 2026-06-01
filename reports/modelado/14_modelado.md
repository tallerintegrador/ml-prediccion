# Entregable 6 — Fase de Machine Learning: costo de servicios logísticos

> Predicción de **costos logísticos de servicios** por operación de importación,
> **antes** de recibir la factura final (pre-liquidación). Target validado con negocio:
> `target_servicios = costo total − tributos`. Tributos se modelan aparte.
> Código: `src/lib/features.py`, `src/ml/08_modelado.py`, `src/ml/09_figuras.py`.
> Artefactos: `models/*.pkl`, `reports/modelado/metrics_*.csv`, `reports/modelado/predicciones_pendientes.csv`.

---

## 1. Diagnóstico breve del estado actual (auditoría previa)

`dataset.load()` reconstruye correctamente el dataset modelable. Verificado:

| chequeo | resultado |
|---|---|
| `load()` construye el modelable | ✔ 1,234 joineadas / **1,058 fiables** / 176 pendientes (`target_fiable`) |
| llave de unión | `op_id_full` presente en todas las tablas, 0 colisiones |
| target servicios bien definido | ✔ `total − tributos`; reconcilia **exacto** con `target_por_concepto` (Σ conceptos≠DERECHOS, dif. ~1e-14) |
| leakage en features | ✔ **ninguna** prohibida en `FEATURES_*` (`fecha_liquidacion`, `fecha_registro_gastos`, `tributos_usd`, `igv_usd`, conceptos, `status` → fuera) |
| features disponibles al predecir | ✔ verificado en las 176 pendientes; **excepciones:** `peso_bruto` (9% disp.) y `pais_origen` (17% disp.) |
| nulos/cardinalidad/outliers | tratados (ver §3): log1p para sesgo, imputación, encoding mixto |

**Hallazgo de calidad:** `peso_bruto` (90.9% nulo, sólo 9% disponible al predecir) y
`pais_origen` (48% nulo, 17% en pendientes) **no son usables de forma fiable** al momento
de la predicción → `peso_bruto` se **descarta**; el origen se cubre vía `pol` (100% disp.).

---

## 2. Dataset final usado para el modelado

- **Filas:** 1,058 operaciones con `target_fiable=True` (liquidadas/archivadas/entregadas).
- **Target:** `target_servicios` (USD). Mediana **$4,790**, media $7,556, skew **8.7** →
  se modela en `log1p`; back-transform con corrección de sesgo log-normal `exp(σ²/2)`.
- **Distribución temporal (año de arribo):** 2020:127 · 2021:261 · 2022:128 · 2023:147 ·
  2024:218 · 2025:176 · 2026:1. Permite forward-chaining limpio.
- **22 operaciones con servicios = 0** (sólo tributos): conservadas (target 0 legítimo).
- **Inferencia futura:** 176 operaciones despachadas sin facturar (mayoría 2022 LIQUIPARCIAL).

### Features (24 entradas → matriz tras encoding) — todas conocidas a la llegada

| bloque | variables | tratamiento |
|---|---|---|
| numéricas sesgadas | `amount_usd`, `qty`, `bulks` | **log1p** + imputar mediana |
| numéricas | `ctnr_qty`, `payment_term`, `transit_days`, `n_oc_distintas`, `n_productos` | imputar mediana (`ctnr_qty`→0 en aéreo) |
| temporales | `anio`, `mes_sin`, `mes_cos` | derivadas de `ata` (mes cíclico) |
| flags | `es_aereo`, `requiere_senasa`, `tiene_seguro` | binarias |
| baja card. (one-hot) | `mode`, `type`, `incoterm`, `categoria_canon`, `customs_agent`, `pod` | one-hot, raras→`infrequent` (min 10) |
| alta card. (target enc.) | `pol`, `supplier`, `shipping_line`, `depot`, `ffw`, `punto_llegada`, `buyer`, `pais_origen` | **target encoding suavizado** (cross-fit, `sklearn.TargetEncoder`) |

**Excluidas:** `peso_bruto` (no disponible al predecir), variables de leakage, conceptos de costo.

---

## 3. Decisiones de limpieza / feature engineering

1. **Sesgo** → `log1p` en `amount_usd`/`qty`/`bulks` (skew 8–9; outliers legítimos no se eliminan, sólo se clipean negativos de captura).
2. **Imputación**: numéricas → mediana del **train del fold**; `ctnr_qty` nulo en aéreo → 0 (faltante estructural, no error); categóricas → `DESCONOCIDO`.
3. **Cardinalidad alta** (`pol` 83, `supplier` 143, `shipping_line` 62…) → **target encoding suavizado con cross-fitting** (evita leakage y sobreajuste); baja card. → one-hot con agrupación de categorías raras.
4. **Categorías raras / deriva semántica**: ya canonizadas en `dataset.load()` (`categoria_canon`, `incoterm_grupo`); one-hot agrupa <10 obs.
5. **Sin leakage en el encoding**: todos los estadísticos (mediana, target-encoding, margen conformal) se ajustan **sólo con años ≤ t** dentro de cada ventana de validación.
6. **Temporales cíclicas**: mes de arribo en `sin/cos`; `anio` como tendencia.

---

## 4. Modelos entrenados y validación

- **Baseline**: mediana de `target_servicios` por segmento `mode × categoria_canon` (fallback: mediana por modo → global).
- **ML**: **LightGBM** (principal), **XGBoost** y **RandomForest** (comparación). Objetivo log1p.
- **Validación TEMPORAL forward-chaining** por año (entrenar ≤ t, validar t+1); **nunca** K-fold aleatorio. 5 ventanas: 2021…2025.

### 4.1 Desempeño global (pooled de todas las ventanas, n=931)

| modelo | MAE | RMSE | WAPE | MdAPE | bias |
|---|---:|---:|---:|---:|---:|
| baseline | 5,315 | 12,850 | 0.652 | 0.510 | −3,786 |
| RandomForest | 5,191 | 13,242 | 0.636 | 0.461 | −2,702 |
| LightGBM | 5,017 | 13,171 | 0.615 | 0.424 | −2,981 |
| XGBoost | 4,936 | 13,126 | **0.605** | **0.422** | −3,370 |

> El pooled está **penalizado por la ventana 2021** (entrenando sólo con 2020 = 127 ops,
> mediana $2.3k → validar 2021, mediana $5.7k): error enorme por **escasez de historia**,
> no por el modelo. Es el piso pesimista.

### 4.2 Desempeño en el **fold de despliegue** (val 2025 — el representativo, n=176)

| modelo | MAE | RMSE | WAPE | MdAPE | bias |
|---|---:|---:|---:|---:|---:|
| baseline | 3,620 | 5,085 | 0.501 | 0.447 | −2,643 |
| RandomForest | 3,232 | 5,866 | 0.448 | 0.324 | +1,713 |
| **LightGBM** | 2,640 | 4,938 | 0.366 | **0.275** | +1,114 |
| **XGBoost** | 2,540 | 4,447 | **0.352** | **0.233** | +553 |

> Con historia suficiente, el ML **reduce el MdAPE de 45% (baseline) a 23–28%** y el WAPE
> de 0.50 a 0.35. Coincide con el `MdAPE 24.9%` reportado en el EDA. Bias casi nulo →
> la corrección de sesgo log-normal funciona. Figuras:
> `reports/modelado/figures/14_modelos_despliegue.png`, `14_mdape_pooled_vs_deploy.png`.

### 4.3 Comparación: modelo directo vs modelo **por concepto**

| enfoque | MAE | RMSE | WAPE | MdAPE |
|---|---:|---:|---:|---:|
| **directo** (LightGBM, total) | 5,017 | 13,171 | **0.615** | **0.424** |
| por concepto (12 LGBM, suma) | 5,624 | 13,708 | 0.690 | 0.556 |

> El modelo **directo gana**: descomponer en 12 conceptos acumula error (cada concepto
> aporta su propio sesgo). El por-concepto sólo conviene si luego se necesita el desglose.

### 4.4 Importancia de features (LightGBM final, top)

`amount_usd` (valor mercadería) ≫ `shipping_line`, `bulks`, `qty`, `supplier`, `pol`,
`ffw`, `depot`, `punto_llegada`, `transit_days`. Confirma los drivers del EDA
(valor + proveedores logísticos + escala). Detalle: `reports/modelado/feature_importance.csv`.

---

## 5. Métricas por segmento (LightGBM, OOF)

`reports/modelado/metrics_por_segmento.csv` (extracto, n≥10):

| mode × categoria | n | WAPE | MdAPE | nota |
|---|---:|---:|---:|---|
| SEA × SUSTRATOS | 255 | 0.68 | 0.45 | el de mayor gasto; cola pesada (RMSE alto) |
| SEA × EMPAQUE | 159 | 0.44 | 0.36 | bien |
| AIR × REPUESTOS | 120 | 0.42 | 0.38 | montos chicos |
| SEA × AGROQUIMICOS | 109 | 0.63 | 0.48 | volátil |
| AIR × PLANTAS | 93 | 0.39 | 0.30 | bien |
| AIR × EMPAQUE | 57 | 0.82 | 0.78 | **débil** (mezcla courier/consolidado) |
| AIR × EQUIPOS_INSTR | 57 | 0.60 | 0.39 | aceptable |

> El error se concentra donde el gasto se concentra (SEA×SUSTRATOS) y en segmentos
> aéreos heterogéneos (AIR×EMPAQUE). Para esos conviene peso bruto / tarifa cotizada.

---

## 6. Intervalos de predicción (P10–P90, conformal)

Regresión cuantílica LightGBM (pinball) **conformalizada (CQR)**: el margen se calibra
con un 30% del train de cada ventana para alcanzar cobertura nominal 80%.

| | cobertura | ancho mediano |
|---|---:|---:|
| cuantil crudo | 0.585 | $3,989 |
| **conformal (CQR)** | **0.760** | $7,114 |

> Cobertura empírica 76% (vs 80% nominal): aceptable bajo deriva temporal. El ancho
> mediano (~$7k) refleja la incertidumbre real de la pre-liquidación. Modelos guardados:
> `models/lgbm_servicios_p10.pkl`, `_p90.pkl` (margen log = 0.141 en `model_meta.json`).

---

## 7. Predicción de operaciones pendientes sin facturar

`reports/modelado/predicciones_pendientes.csv` — **176 operaciones** despachadas sin liquidar:

- Provisión total de servicios prevista: **≈ USD 1,063,197**.
- Desglose por año: 2022 → $971,763 (146 ops) · 2024 → $24,718 (5) · 2025 → $66,716 (25).
- Cada fila trae `servicios_pred_usd`, `p10_usd`, `p90_usd` (el punto cae dentro del intervalo en 96.6%).

---

## 8. Riesgos metodológicos detectados

1. **Deriva temporal volátil** (no un +15%/año limpio): los costos saltan año a año
   (2020→2021 ×2.5). Los árboles **no extrapolan** más allá del último año visto → con poca
   historia (fold 2021) el modelo sub-predice fuerte. Mitiga sólo con más historia, no con
   un factor de tendencia (el slope es ruidoso). **El pooled MdAPE 42% es el piso pesimista;
   el realista es ~23–28%.**
2. **Concentración del gasto**: SEA×SUSTRATOS domina; un sesgo ahí mueve la provisión total.
3. **Cola pesada / outliers legítimos**: RMSE alto en marítimo; WAPE/MdAPE son las métricas de confianza.
4. **Target encoding**: cross-fit en CV evita leakage, pero el modelo final (ajustado en todo
   el fiable) tiene un leve optimismo en categorías raras → vigilar proveedores nuevos.
5. **Features faltantes al predecir** (`peso_bruto`, `pais_origen`): hoy aportan poco porque
   no están disponibles; su ausencia limita el techo de precisión, sobre todo en aéreo.
6. **Pendientes 2022**: 146 de las 176 son de 2022; su predicción usa un modelo entrenado con
   años posteriores (interpolación), razonable, pero su intervalo es más ancho.

---

## 9. Recomendación final — ¿útil para pre-liquidación?

**Sí, con alcance acotado.** Para **provisión agregada** y **estimación por operación con
intervalo**, el modelo es claramente útil: en el escenario de despliegue reduce el error a
**WAPE ~0.35 / MdAPE ~23–28%**, muy por encima del baseline de mediana por segmento
(MdAPE 45%) y del estado actual (provisión manual). Recomendación operativa:

- **Usar el valor puntual** (`servicios_pred_usd`) para provisionar, y **el intervalo P10–P90**
  para gestionar riesgo (provisionar conservador con P90 en segmentos volátiles).
- **Modelo principal: LightGBM** (XGBoost queda como alterno, desempeño equivalente).
- **No** usarlo aún como cifra contable final en segmentos AIR×EMPAQUE / AGROQUIMICOS
  (MdAPE ≥ 0.48): ahí mantener revisión humana.
- Tributos: **modelar aparte** (arancel × CIF), no incluidos en este modelo.

---

## 10. Próximos pasos para mejorar precisión

1. **Peso bruto histórico** (hoy sólo 2025-26): driver clave del costo aéreo y unitario —
   completarlo habilita un modelo de costo por kg, mucho más estable en aéreo.
2. **Partida arancelaria (HS) / tasa Ad Valorem**: haría los **tributos calculables** (59% del costo total) en vez de modelados.
3. **Tarifa de flete cotizada (quote)** al embarque y **tipo de cambio por factura**: convierten el mayor componente de servicios (flete) de "predicho" a "casi conocido".
4. **Nº real de contenedores** y **zona/distancia de fundo destino**: mejoran handling y transporte T2.
5. **Más historia reciente** (re-entrenar trimestralmente) para seguir la deriva de tarifas.
6. **Modelo de tributos** separado (arancel×CIF) y **tuning Optuna** del LightGBM con early stopping temporal.
7. Reincorporar `pais_origen` vía mapa `POL→país` aprendido (imputación) cuando se priorice.

---

## 11. Resultados de la mejora — Fases 1 y 2 (solo modelado, sin datos nuevos)

> Implementadas las acciones del plan `15_plan_mejora.md` que NO requieren datos nuevos.
> Cambios en código: `src/lib/features.py` (+`incoterm_grupo`), `src/ml/08_modelado.py`
> (ratio, smearing robusto, ensamble, tuning Optuna, validación trimestral + multi-semilla).

### 11.1 Qué se hizo

1. **Feature olvidada** `incoterm_grupo` (quién paga el flete) ahora entra al modelo (one-hot).
2. **Modelo de ratio** `servicios/amount_usd`: **probado y DESCARTADO** — los outliers de
   `amount_usd` amplifican el error y el back-transform sobre residuos log-ratio de cola
   pesada dispara la predicción (WAPE>1.3, bias +10k). El **directo gana con claridad**.
3. **Back-transform smearing de Duan** (no paramétrico, winsorizado a [p1,p99]) en lugar de
   la corrección log-normal `exp(σ²/2)`: más robusto a la cola pesada.
4. **Ensamble** `ens_direct` = media de LightGBM + XGBoost (ambos directos, tuneados).
5. **Tuning Optuna** (40 trials, objetivo WAPE forward-chaining) del LightGBM. Mejores
   params: `n_estimators=1200, lr=0.014, num_leaves=87, min_child_samples=10,
   subsample=0.69, colsample=0.70, reg_lambda=6.9, reg_alpha=0.015`.
6. **Validación robusta (Fase 2):** forward-chaining **trimestral** (más folds) y
   **estabilidad multi-semilla** del fold de despliegue.

### 11.2 Desempeño tras la mejora — fold de despliegue (val 2025, n=176)

| modelo | MAE | RMSE | WAPE | MdAPE | bias |
|---|---:|---:|---:|---:|---:|
| baseline | 3,620 | 5,085 | 0.501 | 0.447 | −2,643 |
| RandomForest | 3,010 | 5,519 | 0.417 | 0.308 | +1,388 |
| LightGBM (tuned) | 2,653 | 4,644 | 0.367 | 0.303 | +1,352 |
| XGBoost | 2,543 | 4,502 | 0.352 | 0.250 | +666 |
| **ens_direct (CHAMPION)** | **2,510** | **4,476** | **0.348** | **0.249** | +1,009 |

Pooled (todas las ventanas, n=931): `ens_direct` WAPE **0.593** / MdAPE **0.410**
(antes LightGBM 0.615 / 0.424).

> **Lectura honesta:** la mejora es **real pero modesta**. El WAPE de despliegue baja de
> 0.366 (mejor previo LGBM) a **0.348** (mejor histórico) y el pooled de 0.615 a 0.593; el
> MdAPE de despliegue queda en **~0.25**, igual que el mejor modelo previo (XGBoost 0.233).
> **Conclusión clave:** con los datos actuales el techo está en **MdAPE ~25% / WAPE ~0.35**.
> La meta "90%" (MdAPE ≤10%) **no es alcanzable solo con modelado** — requiere las Fases 3–4
> (peso bruto, tarifa de flete cotizada, partida arancelaria para tributos). Este experimento
> lo confirma cuantitativamente.

### 11.3 Validación robusta (Fase 2)

| esquema | n | WAPE | MdAPE | nota |
|---|---:|---:|---:|---|
| forward-chaining **trimestral** | 992 | 0.523 | 0.385 | más folds; incluye trimestres tempranos difíciles |
| **estabilidad** deploy (5 semillas) | 176 | 0.376 ± 0.010 | 0.299 ± 0.006 | desviación baja → el número **no es frágil** |

> El trimestral (peor que el deploy 2025 porque promedia ventanas con poca historia) y la
> baja desviación entre semillas dan **confianza**: el resultado no depende de un fold ni de
> una semilla afortunada.

### 11.4 Modelo campeón y artefactos

- **Champion = `ens_direct`** (media LightGBM+XGBoost). Supera el pick de modelo único de §4.2/§9.
- Guardado en **`models/champion_servicios.pkl`** (dict picklable: miembros + corrección de
  sesgo + params). `models/lgbm_servicios.pkl` se conserva (LGBM directo, interpretabilidad).
- `model_meta.json` ahora versiona (`version="2-fase1-2"`, champion, params, margen conformal 0.177).
- Provisión de pendientes recalculada: **≈ USD 1,071,448** (176 ops); intervalos P10–P90 con
  cobertura conformal 0.748, ancho mediano ~$7,215.

---

### Cómo reproducir

```bash
venv/Scripts/python.exe src/ml/08_modelado.py   # entrena, valida, predice, guarda artefactos
venv/Scripts/python.exe src/ml/09_figuras.py    # figuras del reporte
```

Artefactos: `models/lgbm_servicios*.pkl`, `models/baseline_segmento.pkl`,
`models/model_meta.json`; métricas en `reports/modelado/metrics_*.csv`,
predicciones en `reports/modelado/predicciones_pendientes.csv`.
