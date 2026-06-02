# Plan de mejora del modelo de costos logísticos

> Objetivo: pasar del estado actual (**deploy MdAPE ~25% / WAPE ~0.35**) hacia el
> mejor error alcanzable, y dejar claro qué hace falta para acercarse al "90%".
> Aterriza sobre el código existente: `src/lib/features.py`, `src/ml/08_modelado.py`,
> `src/lib/dataset.py`.

---

## 0. Reencuadre del objetivo (antes de tocar nada)

Esto es **regresión**, no clasificación: no existe "accuracy 90%". El equivalente honesto:

| "precisión" coloquial | métrica real | estado hoy | meta realista | meta "90%" |
|---|---|---|---|---|
| ~75% | MdAPE 25% | ✔ deploy 2025 | **12–15%** | ≤10% (requiere datos nuevos) |
| — | WAPE 0.35 | ✔ | 0.20 | ≤0.10 |

- **Fases 1–2 (solo modelado):** bajan el error con los datos actuales. Techo estimado ~MdAPE 15–18%.
- **Fases 3–4 (datos nuevos):** únicas que habilitan MdAPE ≤10% ("90%"). Sin ellas no se llega.
- **Fases 5–6:** productización y entrega.

Definir el criterio de éxito **antes** de iterar evita perseguir un número imposible con los datos de hoy.

---

## ✅ Estado de ejecución (2026-06-01) — Fases 1 y 2 IMPLEMENTADAS

Fases 1 y 2 ejecutadas en código (`features.py`, `08_modelado.py`, `09_figuras.py`).
Resultado detallado en `14_modelado.md §11`. Resumen:

| acción | estado | resultado |
|---|---|---|
| 1.1 `incoterm_grupo` + auditoría leakage | ✅ hecho | feature añadida (one-hot) |
| 1.2 modelar ratio servicios/amount | ✅ probado → **DESCARTADO** | WAPE>1.3; outliers de amount lo rompen |
| 1.3 tuning Optuna + early stopping | ✅ hecho | 40 trials; mejores params en meta |
| 1.4 smearing de Duan (robusto) | ✅ hecho | mejora MdAPE LGBM 0.275→0.30 con menos bias |
| 1.5 ensamble LGBM+XGB | ✅ hecho | **champion = `ens_direct`** |
| 2 validación robusta (trimestral + multi-seed) | ✅ hecho | WAPE deploy 0.376 ± 0.010 (estable) |

**Números (fold despliegue 2025):** champion WAPE **0.348** (antes 0.366), MdAPE **0.249**;
pooled WAPE 0.615→**0.593**. Artefacto: `models/champion_servicios.pkl`.

> **Veredicto:** mejora real pero modesta. **El techo con datos actuales queda confirmado en
> MdAPE ~25% / WAPE ~0.35.** La meta "90%" NO se alcanza solo con modelado → quedan las
> Fases 3 y 4 (datos nuevos), que son el único camino real y dependen de negocio.

**Pendiente (requiere datos / negocio):** Fase 3 (peso bruto, quote de flete, TC, contenedores),
Fase 4 (tributos por partida arancelaria), Fase 5 (MLOps), Fase 6 (entrega final).

---

## FASE 1 — Quick wins de modelado (sin datos nuevos) · esfuerzo bajo, impacto medio

Orden por relación impacto/esfuerzo. Todo cabe en `features.py` y `08_modelado.py`.

### 1.1 Usar features ya calculadas pero ignoradas
- `incoterm_grupo` se deriva en `dataset.py` pero **no entra al modelo**. Agregarlo a `CAT_OH` en `features.py`. Define quién paga el flete → driver directo del costo de servicios.
- Revisar si `trimestre_arribo`, `dias_*` aportan **sin leakage**. Ojo: `dias_levante`, `dias_a_fundo`, `costo_unit_*`, `ratio_flete_valor` usan fechas/target posteriores a la llegada → **prohibidos** (leakage). Solo `transit_days` (atd→ata) es legítimo y ya está.

### 1.2 Modelar el costo unitario (ratio), no el monto absoluto  ⟵ mayor impacto esperado
- En vez de predecir `target_servicios`, predecir `ratio = target_servicios / amount_usd`
  (o `/ qty`) y reconstruir `pred = ratio_pred × amount_usd`.
- `amount_usd` se conoce al predecir → **no es leakage**. El ratio tiene mucho menos
  sesgo y cola que el monto absoluto (el EDA ya mostró que `amount_usd` es el driver #1).
- Comparar contra el modelo directo en el mismo CV temporal. Quedarse con el mejor.

### 1.3 Tuning + early stopping (hoy: 500 árboles fijos, lr 0.03, sin tuning)
- Optuna sobre LightGBM con **early stopping temporal**: usar el último año del train
  como watchlist en cada fold (no random). Tunear `num_leaves`, `min_child_samples`,
  `learning_rate`, `reg_lambda`, `n_estimators`.
- Mantener la validación forward-chaining como criterio (no K-fold aleatorio).

### 1.4 Back-transform más robusto
- Hoy: corrección log-normal paramétrica `exp(σ²/2)` (asume residuos normales en log).
- Probar **smearing de Duan** (no paramétrico): `mean(exp(resid))`. Más robusto a cola
  pesada; comparar bias en deploy.

### 1.5 Ensamble LGBM + XGBoost
- En deploy ambos empatan (WAPE 0.366 vs 0.352). Promediar sus predicciones suele
  recortar varianza sin coste. Probar promedio simple y promedio ponderado por WAPE OOF.

### 1.6 Tratamiento de segmentos débiles
- `AIR×EMPAQUE` (MdAPE 0.78) y `AGROQUIMICOS` (0.48) arrastran el error. Opciones:
  flag de courier vs consolidado, o modelo/segmento aparte. Mantener revisión humana ahí.

**Entregable Fase 1:** rerun de `08_modelado.py`, tabla comparativa antes/después en
`14_modelado.md`, y decisión de campeón (directo vs ratio, single vs ensamble).

---

## FASE 2 — Validación y confianza · esfuerzo bajo, impacto en credibilidad

- **Más ventanas:** además del forward-chaining anual, probar ventanas **trimestrales**
  para tener más folds tipo-deploy y un MdAPE menos ruidoso.
- **Repetir con varias semillas** y reportar media ± desviación (hoy una sola semilla).
- **Nested CV** para que el tuning de Fase 1.3 no contamine la métrica de validación.
- **Calibración del intervalo:** cobertura conformal hoy 76% vs 80% nominal. Subir el
  nivel de calibración o el % de calibración para cerrar la brecha.

**Entregable Fase 2:** métricas con intervalos de confianza; el número que se reporta
deja de ser un punto frágil.

---

## FASE 3 — Datos nuevos (el verdadero techo hacia "90%") · esfuerzo alto, impacto alto

Sin esto, MdAPE ≤10% **no es alcanzable**. Orden por impacto:

1. **Peso bruto histórico** (hoy 91% nulo, ya descartado en `features.py`). Habilita un
   **modelo de costo por kg** — mucho más estable, sobre todo en aéreo (donde más falla).
2. **Tarifa de flete cotizada (quote) al embarque.** El flete es el mayor componente de
   servicios; con la cotización pasa de "predicho" a "casi conocido".
3. **Tipo de cambio por factura** y **nº real de contenedores** (hoy `ctnr_qty` con huecos).
4. **Zona/distancia del fundo destino** → mejora handling y transporte T2.
5. **`pais_origen` por mapa POL→país aprendido** (imputación): recupera la feature que hoy
   tiene 48% de nulos. Quick-ish: se puede empezar con un diccionario POL→país.

**Entregable Fase 3:** checklist de datos a pedir a negocio/operaciones, con el impacto
esperado de cada uno. Es una conversación de negocio, no solo de código.

---

## FASE 4 — Modelo de tributos separado · esfuerzo medio, impacto alto en el costo TOTAL

- Tributos = **59% del costo total** y hoy quedan fuera (se modelan "aparte" pero no existen).
- Con **partida arancelaria (HS) + tasa Ad Valorem**, los tributos son **calculables**
  (`Ad Valorem × CIF`), casi sin error, en vez de predichos.
- Sumar `servicios_pred + tributos_calc` da un `target_total` mucho más preciso que
  cualquier modelo directo sobre el total.

**Entregable Fase 4:** módulo `src/ml/tributos.py` (cálculo aranceles) + integración en
la predicción de pendientes.

---

## FASE 5 — Productización / MLOps · esfuerzo medio, impacto operativo

- **Re-entrenamiento trimestral** automatizado (la deriva de tarifas es fuerte, ×2.5 año a año).
- **Monitoreo de drift** de features y de error en producción.
- **Versionado de modelo** (`model_meta.json` ya guarda metadatos; añadir versión + fecha + métricas).
- **Interfaz de consumo:** predicción batch sobre nuevas operaciones (ya existe
  `predicciones_pendientes.csv`) + función de scoring única documentada.

---

## FASE 6 — Documentación y entrega

- Actualizar `14_modelado.md` con resultados de Fases 1–2.
- Dejar explícito en el reporte: **el modelo sirve para provisión agregada + intervalo
  P10–P90 por operación**, no como cifra contable puntual exacta en segmentos volátiles.
- Cerrar el relato del "90%": qué se logró con datos actuales y qué datos lo desbloquean.

---

## Resumen ejecutivo (orden de ataque)

| # | Acción | Esfuerzo | Impacto en error | Habilita "90%" |
|---|---|---|---|---|
| 1.2 | Modelar ratio costo/valor | bajo | **alto** | no, pero acerca |
| 1.3 | Tuning + early stopping | bajo | medio | no |
| 1.1 | Usar `incoterm_grupo` y limpiar leakage | muy bajo | bajo-medio | no |
| 1.4–1.5 | Smearing + ensamble | bajo | bajo-medio | no |
| 2 | Validación robusta / IC | bajo | credibilidad | no |
| 4 | Tributos calculables (HS) | medio | **alto (total)** | sí (en total) |
| 3 | Peso bruto, quote, TC, contenedores | alto | **alto** | **sí** |
| 5–6 | MLOps + entrega | medio | operativo | — |

**Camino corto (solo código): ✅ EJECUTADO.** Fases 1 + 2 dieron WAPE 0.366→**0.348** /
MdAPE ~0.25. La meta optimista de 15–18% **no se alcanzó**: el techo con datos actuales es
~25%, confirmado empíricamente. Ganancia real pero marginal; el valor mayor fue cerrar la
incertidumbre (validación robusta) y descartar el ratio.
**Camino al "90%" (requiere negocio): PENDIENTE.** Fases 3 + 4 → datos nuevos = único techo real.
