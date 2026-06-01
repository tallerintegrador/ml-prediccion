# Entregable 3 — Reporte de hallazgos del EDA v2
### Sistema predictivo de costos de importación — Hortifrut Perú

> Reconstruido **desde cero** a partir de los 8 CSV crudos (`data_csv/raw/`).
> Target validado con negocio: **servicios logísticos** (tributos modelados aparte),
> sobre operaciones con **target fiable** (liquidadas/archivadas/entregadas).
> Pipeline reproducible en `src/00…07`; reportes detallados en `reports/00…12`.

---

## 1. Resumen ejecutivo

Se descubrieron **dos familias de archivos**: **7 reportes operativos** (1 fila por
OC/producto, con deriva de esquema español→inglés entre 2019 y 2026) y **1 expense
report** (14,879 líneas de costo, 1 fila por concepto facturado). Tras armonizar ~65
variables canónicas y canonicalizar 81 conceptos de costo en 14 grupos, se reconstruyó
el target uniendo ambas familias por el **identificador completo de operación**
(`op_id_full`), con **0 colisiones** y **98.4% de cobertura** del expense
(**1,234 operaciones** con costo + datos operativos; 196 despachadas aún sin facturar =
el escenario real de predicción).

El **costo logístico de servicios** por operación tiene **mediana ≈ $4.9k**, es fuerte­
mente asimétrico (skew≈9) y se normaliza bien en **log**. Los **drivers confirmados con
evidencia** (Kruskal-Wallis + ε²) son, en orden: **naviera/depósito, tipo de carga
(FCL/LCL/aéreo), categoría de producto, modo, agencia de aduana e incoterm**; y, en
escala, **nº de bultos** (Spearman 0.77) y **valor de mercadería** (0.63). El gasto se
**concentra**: SUSTRATOS×marítimo = 46% del total de servicios. Hay **deriva temporal
≈ +15%/año** en el costo unitario → obliga a **validación temporal**. Un LightGBM
baseline (split temporal, sin tuning) ya da **MdAPE 24.9%**, confirmando señal predictiva.

---

## 2. Estructura descubierta de los 8 archivos

| familia | archivos | granularidad | rol |
|---|---|---|---|
| Operativo | `report_importaciones_{2019-2020…2026}` (7) | 1 fila / OC-producto-contenedor | features logísticas |
| Costos | `bd_expense_report_importaciones_201X` (1) | 1 fila / concepto facturado | **target** |

- **Deriva de esquema** confirmada: columnas cambian de idioma (CAMPAÑA→SEASON,
  ESTADO→STATUS, CATEGORIA→CATEGORY, AG. ADUANA→CUSTOMS AGENT…) y el separador de 2021
  es `;`. Resuelta con un mapa canónico (`reports/02_armonizacion_columnas.md`).
- **Llave de unión:** `op_id_full` (ID normalizado, p. ej. `20119HPER`). La llave laxa
  `YY-NNN` colapsaba 108 operaciones distintas → descartada.

## 3. Target: localización y reconstrucción

- Vive en el expense: **`Monto Total USD`** (ya convertido USD/EUR/PEN→USD, **pre-IGV**).
- **81 conceptos → 14 canónicos** (sólo 0.1% en "otros"). Composición: DERECHOS/IMPUESTOS
  59%, IGV 11% (excluido), FLETE 10%, DESCARGA 5%, TRANSPORTE T2 5%, etc.
- **Target servicios** = Σ Monto Total USD por operación − tributos − IGV. **Tributos**
  (Ad Valorem/derechos, 59%) se modelan aparte (lógica arancel×CIF).
- Distribución servicios: media $7.7k, mediana $4.9k, p95 ~$30k, skew≈9 → **log / Gamma-Tweedie**.

## 4. Calidad de datos

- 46 columnas usables (<30% nulos) / 17 condicionales / 7 descartables.
- Ausencia **estructural** (no error): `ctnr_qty` nulo 100% en aéreo vs 10% marítimo;
  `peso_bruto` y `country_origin` sólo desde años recientes.
- Inconsistencias menores: 9 tránsitos imposibles (ata<atd), 77 levantes pre-arribo,
  `insurance_hper` con una fecha colada, deriva semántica en `categoria` (resuelta → 8 grupos).

## 5. Las 9 preguntas (sección 6)

**1) ¿Qué contiene cada archivo y cómo se relacionan?**
7 operativos (features, 1 fila/OC-producto) + 1 expense (costos, 1 fila/concepto). Se unen
por `op_id_full`; el expense aporta el target y los operativos las features. 1,234 operaciones
cruzan (98.4%); 196 operativas sin costo aún (predicción) y 42 expense huérfanas (legacy 2019).

**2) ¿Dónde está el target y cómo se reconstruye?**
En `Monto Total USD` del expense. Se agrupa por operación y se suma, **excluyendo
Percepción del IGV** y separando los **tributos**. Resultado: `target_servicios` (primario)
y `tributos_usd` (modelo aparte).

**3) ¿Distribución y transformación del target?**
Asimetría positiva fuerte (skew≈9; cola hasta $241k). **log1p** la normaliza
(Shapiro mejora varios órdenes). Recomendado: log-target con corrección de sesgo, o
objetivo **Gamma/Tweedie**.

**4) ¿Los 5–8 drivers principales (con evidencia)?**
Por ε² (Kruskal-Wallis): **shipping_line 0.47**, **depot 0.46**, **type 0.45**,
**categoria 0.41**, **mode 0.34**, **customs_agent 0.30**, **incoterm 0.29**. Por escala
(Spearman): **bulks 0.77**, **amount_usd 0.63**, **peso_bruto 0.52**. Confirmado por la
importancia LightGBM (bulks, amount_usd, categoría, shipping_line, incoterm_grupo).
Hallazgo corrector: **SEA cuesta más por operación** ($6.9k vs $1.3k aéreo) — el "aéreo
es más caro" sólo aplica por kg, no por operación.

**5) ¿Variables con leakage?**
Conceptos de costo desglosados (son el target), `fecha_liquidacion`,
`fecha_registro_gastos`, Nº de liquidación/documento, el propio `status` de liquidación,
e `igv_usd/tributos_usd` como inputs del modelo de servicios. Detalle en `10_diagnostico.md`.

**6) ¿Qué calidad/datos faltan?**
Falta **peso bruto histórico** (sólo 2025-26), **partida arancelaria/tasa Ad Valorem**
(haría los tributos calculables), **tipo de cambio por factura**, **tarifa de flete
cotizada** y **nº real de contenedores**. Recopilarlos elevaría la precisión.

**7) ¿Volumen suficiente / riesgo de sobreajuste?**
**1,058 operaciones fiables** alcanzan para árboles con regularización (LightGBM) pero
**no** para one-hot masivo de alta cardinalidad → usar target/frequency encoding y CV
temporal. Riesgo moderado; reportar error por segmento (el gasto se concentra).

**8) ¿Validación y métricas?**
**Forward chaining por campaña** (entrenar ≤año t, validar t+1); **nunca** K-fold aleatorio
(deriva +15%/año). Métricas: **MAE** y **WAPE** (provisión total), **MdAPE** (robusta),
**RMSE**, y **pinball loss + cobertura de intervalos** para la pre-liquidación con incertidumbre.

**9) ¿Features y algoritmo recomendados?**
Features por bloque en `11_features_candidatas.md` (mercadería, transporte/modo, ruta,
contractual, proveedores logísticos, temporales). Algoritmo: **LightGBM** (maneja nulos
y categóricas, robusto a outliers, captura no-linealidades) con target log/Tweedie y
encoding mixto; baseline ya en **MdAPE 24.9%**. Cuantiles (P10/P90) para intervalos.

---

## 6. Próximos pasos (Fase de modelado)
1. Feature engineering completo (zona_ruta, imputación de país desde POL, peso retro).
2. Encoding (one-hot + target encoding suavizado) y pipeline de imputación.
3. LightGBM con validación forward-chaining + Optuna; modelos separados servicios / tributos.
4. Intervalos por regresión cuantílica (pinball) y reporte de error por segmento.
