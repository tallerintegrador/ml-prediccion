# Prompt para la fase de MODELADO — Sistema predictivo de costos de importación (Hortifrut Perú)

> Copia todo lo que sigue (desde "## 1. Rol" hasta el final) en un chat nuevo para arrancar el modelado.

---

## 1. Rol e instrucción general

Actúa como un **Ingeniero Senior de Machine Learning** especializado en logística e importaciones. Vas a ejecutar la fase de **Modeling** de CRISP-DM sobre datos históricos de importaciones de Hortifrut Perú. **El EDA ya está hecho** (fase previa, rama `feature/version_2`): no lo repitas; parte de sus resultados procesados y construye el sistema de predicción.

**Objetivo:** predecir el **costo logístico de servicios en USD por operación de importación**, en el momento en que la mercancía llega (antes de recibir las facturas de los proveedores logísticos), para que Contabilidad pre-liquide y provisione el gasto en SAP con un intervalo de incertidumbre.

Sé altamente técnico, cuantitativo y reproducible. Trabaja en **español**, tono de consultor técnico.

---

## 2. Contexto de negocio

Hortifrut Perú (agroexportadora) importa plantas/plantines, sustratos (fibra de coco, corteza de pino), insumos de empaque (film, punnet), agroquímicos y equipos/repuestos de riego, desde Chile, España, EE.UU., Países Bajos, Bélgica, China, etc., por vía **aérea** y **marítima**. Cadena: origen → naviera/aerolínea → puerto embarque → tránsito → Callao/Lima → agenciamiento aduana + DAM → inspección/liberación SENASA → levante → depósito temporal → transporte a Lima → transporte a fundo → recepción. Las facturas logísticas llegan tarde y bloquean la mercancía; hoy la pre-liquidación es manual. El modelo debe estimar ese costo apenas llega la carga.

---

## 3. Estado actual (qué ya existe — NO rehacer)

Proyecto en `c:\Users\lucia\OneDrive\Documents\HortifrutCostosImport`. Entorno: `venv/` con Python 3.14, **pandas 3.0, numpy 2.4, scikit-learn 1.8, lightgbm 4.6, xgboost 3.2, scipy 1.17, matplotlib, seaborn, pyarrow**. NO están instalados `shap` ni `ydata_profiling` (instálalos si los necesitas). El trabajo v1 está en `legacy/` (referencia; NO reutilizar sus artefactos directamente).

Pipeline del EDA v2 ya construido en `src/`:
- `config.py` — rutas, mapa canónico de columnas, reglas de conceptos, metadatos por variable (rol/leakage).
- `utils.py` — parsers de montos/fechas/nulos, normalización, llaves de operación, conceptos canónicos.
- `dataset.py` — **carga preparada**: `dataset.load(fiable_only=True)` devuelve el dataframe modelable con targets, armonización categórica y features derivadas. Reúsalo.
- `00_inventario … 07_diagnostico` + `run_all.py`.

**Datos procesados listos (úsalos como punto de partida):**
- `data_csv/processed/operaciones_modelables.parquet` — **1,234 operaciones × 73 col** (1 fila/operación, con target).
- `data_csv/processed/target_por_concepto.parquet` — **pivot operación × 14 conceptos canónicos** (USD por concepto). Base para el modelo a nivel concepto.
- `data_csv/processed/expense_lineas.parquet` — 14,879 líneas de costo (1 fila/concepto facturado).
- `data_csv/processed/operativo_lineas.parquet` — 3,088 líneas operativas armonizadas.

Reportes del EDA en `reports/00…13.md` (lee `13_hallazgos.md`, `11_features_candidatas.md`, `12_tabla_decisiones.md`, `05_diccionario_datos.md`).

---

## 4. Decisiones ya validadas con negocio (respétalas, no las rediscutas)

1. **Target primario = servicios logísticos** (`target_servicios` = Σ Monto Total USD − tributos − IGV). Mediana ~$4.9k, media ~$7.7k, cola hasta $241k, **skew≈9** → modelar en **log1p** o con objetivo **Gamma/Tweedie**, con corrección de sesgo al des-transformar.
2. **Tributos (Ad Valorem/derechos, 59% del gasto) se modelan APARTE** — siguen lógica arancel×CIF, no de servicios. NO usar `tributos_usd`/`igv_usd` como features del modelo de servicios (leakage).
3. **Entrenar/evaluar solo sobre `target_fiable=True`** (operaciones liquidadas/archivadas/entregadas): **1,058 ops**. Las 162 parciales (LIQUIPARCIAL) pueden incorporarse con **pesos<1 o como censura** (cota inferior).
4. **Llave de operación = `op_id_full`** (0 colisiones, une 98.4% del expense). Las **196 operaciones despachadas sin facturar** son el **set real de predicción** (no tienen target).
5. **Validación TEMPORAL (forward chaining), nunca K-fold aleatorio** — hay deriva de tarifas +15%/año.
6. Formato: **scripts .py reproducibles en `src/` (fuente de verdad) + notebook de presentación**.

---

## 5. Resultados e interpretaciones del EDA (úsalos como conocimiento previo)

**Distribución del target** (`reports/figures/f02_target_dist.png`, `f04_target.png`): asimetría positiva fuerte; `log10` lo normaliza bien (centrado en ~10⁴); hay un cluster menor de operaciones diminutas ($10–100, courier aéreo/muestras). Servicios y tributos correlacionan moderado (los tributos escalan con el valor de mercadería, ρ≈0.82 con `amount_usd`).

**Composición del costo** (`f02_conceptos.png`): DERECHOS/IMPUESTOS 59%, Percepción IGV 11% (excluido), FLETE_INTERNACIONAL 10%, DESCARGA 5%, TRANSPORTE_T2_LIMA_FUNDO 5%, SOBRESTADIA_ALMACENAJE 3%, resto <2% c/u (handling, agenciamiento, inspección, SENASA, seguros).

**Drivers categóricos** (Kruskal-Wallis + ε², `f05_drivers_cat.png`) — todos efecto grande salvo país (medio) y canal (nulo):
`shipping_line` 0.47 · `depot` 0.46 · `type` 0.45 · `categoria_canon` 0.41 · `mode` 0.34 · `customs_agent` 0.30 · `incoterm` 0.29 · `incoterm_grupo` 0.24 · `ffw` 0.21 · `pais_origen` 0.11 · `canal` ~0.
Interpretación de las cajas: `type` sube monótono AIR < LCL < FCL/LCL < FCL/FCL < FCL; `categoria` sube MUESTRAS < EQUIPOS/REPUESTOS < PLANTAS/AGROQUÍMICOS < EMPAQUE < SUSTRATOS; `shipping_line`/`depot` muestran gradiente courier barato (UPS/FedEx/DHL) → naviera/depósito caro.

**Drivers de escala** (`f05_escala.png`, `f05_corr.png`): **`bulks` ρ=0.77** (el más fuerte), `amount_usd` ρ=0.63 (Pearson-log 0.74), `peso_bruto` 0.52 (pero solo n=96, falta histórico), `qty` 0.44, `transit_days` 0.35, **`ctnr_qty` ρ=0.03 (débil/ruidoso)**. Multicolinealidad a vigilar: `amount_usd`~`peso_bruto` 0.64, `bulks`~target 0.72, `peso`~`transit` 0.69.

**Hipótesis aéreo vs marítimo (corregida):** en términos absolutos por operación **el marítimo es más caro** (servicios mediana SEA $6,854 vs AIR $1,257) por contenedores/depósito/transporte; el aéreo solo es más caro *por kg*. Modelar modo explícitamente.

**Segmentos** (`f07_segmentos.png`): el gasto se concentra — **SUSTRATOS×SEA 46%**, EMPAQUE×SEA 22%, AGROQUIMICOS×SEA 9%, EMPAQUE×AIR 7%, PLANTAS×AIR 7%. Prioriza precisión en estos; los raros (MAQUINARIA n=30, MUESTRAS n=15) serán flojos hasta tener más datos.

**Clusters** (KMeans k=4, silhouette 0.44, `f07_clusters.png`): (0) courier aéreo diminuto n=102 mediana $71; (1) marítimo voluminoso n=616 mediana $7,077 (núcleo); (2) aéreo plantas/repuestos n=328 mediana $1,686; (3) marítimo sustratos outliers n=12.

**Tiempos y estacionalidad** (`f06_temporal.png`): tránsito mediana 11d (aéreo días, marítimo semanas), días en depósito mediana 6d. **Días en depósito ↔ costo de sobrestadía: ρ=0.53** (con sobrestadía: 13d depósito vs 5d sin). Tendencia del costo unitario **+15%/año, p=2e-07** → recencia importa.

**Baseline LightGBM** (split temporal honesto train≤2023 n=663 / test≥2024 n=395, sin tuning): **MAE $3,468 · MdAPE 24.9% · WAPE 41.8%**. Importancia por permutación (`f07_importancia.png`): `bulks` >> `amount_usd` > `categoria_canon` ≈ `shipping_line` > `incoterm_grupo` > `ctnr_qty`. Hay señal clara; queda margen amplio con feature engineering, encoding correcto y modelado a nivel concepto.

---

## 6. Features candidatas (conocidas a la llegada — sin leakage)

Detalle en `reports/11_features_candidatas.md` y `12_tabla_decisiones.md`. Bloques:
- **Mercadería:** `amount_usd` (log), `qty` (log), `categoria_canon` (one-hot 8).
- **Transporte/modo:** `mode`, `type`, `es_aereo`, `ctnr_qty` (0 en aéreo), `bulks` (log), `peso_bruto` (si se completa histórico).
- **Ruta/origen:** `pol` (freq/target enc., 86), `pod` (one-hot, 28), `pais_origen` (imputar desde POL), `zona_ruta` (derivar POL→POD).
- **Comercial:** `incoterm`, `incoterm_grupo`, `payment_term`, `supplier` (target enc. suavizado, 174).
- **Proveedores logísticos:** `shipping_line` (target/freq enc.), `ffw`, `customs_agent`, `depot`.
- **Temporales:** `transit_days`, `mes_arribo` (cíclica), `anio` (validación), `canal_rojo`, `requiere_senasa`, `tiene_seguro`.

**EXCLUIR por leakage:** conceptos de costo desglosados (son el target), `tributos_usd`/`igv_usd`, `fecha_liquidacion`, `fecha_registro_gastos`, Nº liquidación/documento, `status` de liquidación, y fechas posteriores a la llegada (levante/SENASA/recepción) salvo como previsión.

---

## 7. Tarea de modelado (ejecútala en orden)

### 7.1 Arquitectura recomendada — modelo a nivel CONCEPTO
En vez de un solo modelo a nivel operación (1,058 filas), construye un **modelo multi-salida por los 14 conceptos canónicos** y agrega: `costo_total_servicios = Σ conceptos`. Aprovecha `target_por_concepto.parquet` (~9,758 observaciones operación×concepto). Justificación: ~9× más señal, drivers más limpios por componente (flete↔modo/ruta; sobrestadía↔días depósito; descarga↔contenedores), más interpretable. Compáralo contra un **modelo directo a nivel operación** (baseline) y reporta cuál gana por WAPE/MdAPE.

### 7.2 Preprocesamiento
- Target en `log1p` (o Gamma/Tweedie). Corrección de sesgo al des-transformar (smearing de Duan o equivalente).
- Encoding: **one-hot** baja cardinalidad (mode, type, incoterm, categoria, pod); **target/frequency encoding suavizado** (bayesiano, ajustado solo en train de cada fold) para alta cardinalidad (supplier, pol, shipping_line, depot, ffw). Cuidado con leakage del target encoding en la validación temporal.
- Imputación: numéricas → mediana (por modo donde aplique); `ctnr_qty`→0 en aéreo; categóricas → 'DESCONOCIDO'.
- Incorpora las 162 parciales con peso<1 (o censura); pondera por recencia (decaimiento por antigüedad por la inflación +15%/año).

### 7.3 Validación
- **Forward chaining / ventana expansiva por campaña:** entrenar ≤t, validar t+1, para t = 2021…2025. Reporta media e IC de las métricas across folds.
- **Métricas:** MAE y **WAPE** (negocio: provisión total), **MdAPE** (robusta), RMSE, y **pinball loss + cobertura de intervalos**.
- **Reporta error por segmento** (modo×categoría) y por rango de costo; no escondas el peor desempeño en segmentos raros.

### 7.4 Modelos
- Principal: **LightGBM** (maneja nulos/categóricas, robusto a outliers). Tuning con Optuna sobre la CV temporal.
- Compara con un baseline simple (mediana por segmento) y opcionalmente XGBoost.
- **Intervalos de incertidumbre:** regresión cuantílica (P10/P50/P90) o conformal prediction; valida cobertura ≈ nominal.
- **Modelo de tributos aparte:** estima Ad Valorem/derechos (idealmente con tasa arancelaria×CIF; si no hay HS code, modelo ML sobre `amount_usd`/categoría/país).
- Costo total provisionado = servicios (modelo) + tributos (modelo/fórmula).

### 7.5 Producción y reentrenamiento
- Función de predicción que tome una operación recién llegada (features sin leakage) → costo estimado + intervalo.
- Diseña el **ciclo de reentrenamiento**: append de campañas nuevas (~200–270 ops/año) → re-armonizar (el pipeline ya absorbe deriva de esquema) → reentrenar → evaluar contra la campaña más reciente → **monitoreo de drift (PSI)**.
- Predice y entrega el costo de las **196 operaciones despachadas sin facturar** como demostración.

---

## 8. Requisitos técnicos
- Scripts numerados nuevos en `src/` (p. ej. `08_features.py`, `09_modelo_servicios.py`, `10_modelo_tributos.py`, `11_intervalos.py`, `12_demo_prediccion.py`), integrados a `run_all.py`. Reutiliza `config.py`/`utils.py`/`dataset.py`.
- Fija semillas (`config.SEED=42`). Guarda modelos en `models/` (joblib) y métricas/figuras en `reports/`.
- Código limpio, comentado, en español. Cada decisión justificada con números.
- Consola Windows = cp1252: escribe reportes a archivos UTF-8 (usa `utils.MdReport`), no imprimas caracteres especiales a stdout.

---

## 9. Entregables
1. Dataset de features final (parquet) + pipeline de preprocesamiento reproducible.
2. Modelo(s) entrenados (servicios a nivel concepto + tributos) con sus métricas por fold y por segmento.
3. Intervalos de predicción con cobertura validada.
4. Función/demo de predicción para una operación nueva + predicción de las 196 pendientes.
5. Reporte de modelado (resumen ejecutivo + detalle): comparación de arquitecturas, métricas, importancia/SHAP, error por segmento, diseño de reentrenamiento y monitoreo de drift.

## 10. Preguntas a responder al final
1. ¿El modelo a nivel concepto supera al directo por operación? ¿En cuánto (WAPE/MdAPE)?
2. ¿Qué métricas alcanza por segmento y dónde falla (y por qué)?
3. ¿Las features de mayor impacto coinciden con los drivers del EDA?
4. ¿La cobertura de los intervalos es fiable para la pre-liquidación?
5. ¿Cómo se comporta ante la deriva temporal y cómo se mitigará en reentrenamientos?
6. ¿Qué datos adicionales elevarían más la precisión (peso bruto histórico, HS code/arancel, tarifa cotizada)?
