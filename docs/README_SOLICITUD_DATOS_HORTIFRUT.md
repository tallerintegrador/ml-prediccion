# Solicitud de datos para elevar la predicción de costos de importación al 90 %

**Para:** Hortifrut — Operaciones / Comercio Exterior / Sistemas
**De:** Equipo de Modelado (Taller Integrador I — UPAO)
**Asunto:** Datos faltantes necesarios para que el modelo de predicción de costos alcance un error ≤ 10 % ("90 % de precisión")
**Fecha:** 2026

---

## 1. Resumen ejecutivo

Construimos un modelo de Machine Learning que predice, **al momento en que la mercadería arriba al puerto y antes de recibir las facturas**, cuánto costarán los **servicios logísticos** de cada operación de importación (flete, descarga, transporte interno, handling, sobrestadía, agenciamiento, etc.).

Con los datos disponibles hoy el modelo alcanza un **error mediano del ~25 %** (validación honesta, sin trampas). Esto **ya está al límite de lo que los datos actuales permiten**: no es un problema del algoritmo ni de la limpieza de datos, sino de **qué información se registra**.

**El salto al 90 % (error ≤ 10 %) depende de que Hortifrut entregue cinco bloques de datos** que hoy no se capturan o se capturan a medias. Este documento los detalla, con su formato, su fuente y el impacto esperado.

> **Conclusión en una línea:** los datos actuales están limpios y el análisis es correcto; lo que falta es registrar **las variables que realmente determinan el costo**. Sin ellas, ~25 % es el techo físico.

---

## 2. Qué significa "90 %" en este problema

Esto es un problema de **regresión** (predecir un monto en USD), no de clasificación. No existe una "exactitud del 90 %" como en un sí/no. El equivalente honesto y medible es:

- **MdAPE ≤ 10 %** — el error mediano por operación es como máximo 10 %.
- **WAPE ≤ 0.10** — el error agregado sobre el total de dinero es como máximo 10 %.

Hoy estamos en **MdAPE ≈ 25 % / WAPE ≈ 0.35**. La meta es bajar ambos a ≤ 10 %.

---

## 3. Estado actual (validado)

| Métrica | Hoy | Meta |
|---|---:|---:|
| Operaciones con costo real usadas para entrenar | 1,058 | — |
| Error mediano por operación (MdAPE) | **24.9 %** | ≤ 10 % |
| Error agregado (WAPE) | **0.348** | ≤ 0.10 |
| Operaciones ya dentro de ±10 % | 24 % | mayoría |

La validación es **temporal** (se entrena con el pasado y se predice un año que el modelo nunca vio), por lo que el número es realista y no optimista.

---

## 4. Por qué el modelo no puede mejorar con los datos actuales

Hicimos tres comprobaciones para descartar que el problema fuera "mal modelo" o "datos sucios":

### 4.1 El análisis es correcto
La validación temporal reproduce de forma estable el 25 % de error; no hay fuga de información ni atajos.

### 4.2 Operaciones idénticas cuestan hasta 4.6× distinto
Agrupamos operaciones que el modelo ve como **idénticas** (mismo modo de transporte, tipo de carga, incoterm, categoría de producto, puerto de origen, puerto de destino **y el mismo número de contenedores**) y medimos cuánto varía su costo real:

- Dispersión mediana **dentro** de cada grupo: **±36 %**
- Relación entre el costo más caro y el más barato (mediana): **4.6×**

Ejemplos reales (entrada idéntica para el modelo, costo final muy distinto):

| Ruta | Nº ops | Costo mínimo | Costo mediano | Costo máximo |
|---|---:|---:|---:|---:|
| Colombo → Callao, 1 contenedor, sustratos | 67 | $1,704 | $6,533 | $11,743 |
| Valencia → Callao, 1 contenedor, sustratos | 55 | $2,433 | $13,517 | $42,409 |
| Riga → Callao, 1 contenedor, sustratos | 29 | $2,028 | $4,683 | **$129,167** |

> **Interpretación:** dos contenedores que para el sistema son iguales pueden costar uno $1,704 y otro $11,743. El modelo, al recibir la misma entrada, no puede dar dos salidas distintas. La diferencia la explican factores que **hoy no se registran**: días de sobrestadía, el monto del flete cotizado, el peso real, servicios extraordinarios. Ese ~36 % de dispersión es el **piso de error** mientras esos datos no existan.

### 4.3 Faltan las columnas que deciden el costo
Inspeccionando los archivos crudos (`data_csv/raw`):

| Dato que mueve el costo | Estado actual |
|---|---|
| **Partida arancelaria (HS)** — define los tributos (65 % del costo total) | **No existe ninguna columna** |
| **Monto del flete cotizado** — el mayor gasto de servicios | Solo existe la *fecha* de cotización (`QUOTE REQUEST DATE`), no el **monto** |
| **Peso bruto (kg)** | La columna existe pero **solo el 9 % está lleno** |
| **Nº de contenedores** y **país de origen** | **~52 % lleno** (la mitad vacía) |
| **Días reales de sobrestadía / demora** | No se calcula por operación |

---

## 5. Anatomía del costo: dónde está el dinero

Saber qué pesa más permite priorizar. Sobre las 1,058 operaciones fiables, el costo total se reparte así:

| % del total | Concepto | Naturaleza |
|---:|---|---|
| **65.0 %** | Derechos / tributos | **Cálculo exacto** (Ad Valorem × CIF) — hoy fuera del modelo |
| 12.6 % | Flete internacional | Cotizable (el monto se conoce en el booking) |
| 5.8 % | Descarga | Tarifa por contenedor |
| 5.7 % | Transporte puerto → fundo (T2) | Distancia × tarifa |
| 3.2 % | Sobrestadía / almacenaje | Estocástico (demoras) |
| 2.3 % | Servicios logísticos extraordinarios | Variable |
| 1.4 % | Transporte Callao → Lima (T1) | Tarifa |
| 1.2 % | Handling / THC | Tarifa naviera |
| 1.1 % | Agenciamiento aduana | Tarifa fija |
| 1.0 % | Inspección / verificación | Tasa |
| < 1 % | SENASA, seguros, otros | Tasas fijas |

La mayor parte del costo **no es aleatoria: es cálculo y tarifa**. Hoy el modelo *predice* cosas que en realidad son *fórmulas y lookups* porque no tiene los insumos para calcularlas.

---

## 6. Datos solicitados (priorizado)

Esta es la solicitud central. Cada bloque indica **qué pedir, en qué formato y qué desbloquea**.

### Prioridad 1 — Partida arancelaria (HS) y tasa Ad Valorem
> Convierte el 65 % del costo total (tributos) de "estimado" a **exacto**.

| Dato | Formato | Fuente |
|---|---|---|
| Partida arancelaria HS (10 dígitos) por producto | Código | SUNAT / declaración DAM |
| Tasa Ad Valorem por HS | % | Arancel SUNAT |

### Prioridad 2 — Monto del flete cotizado
> El flete es el 36 % de los servicios. Hoy se predice mal; con el monto se vuelve **conocido**.

| Dato | Formato |
|---|---|
| Tarifa de flete cotizada al booking | USD por contenedor (marítimo) o USD/kg facturable (aéreo/LCL) |
| Recargos (BAF / CAF / THC origen-destino) | USD |

*(La fecha de cotización ya se registra; falta capturar el **importe**.)*

### Prioridad 3 — Drivers físicos completos
> Habilitan el costo variable real (descarga, handling, transporte, aéreo).

| Dato | Estado hoy | Acción |
|---|---|---|
| **Peso bruto (kg)** | 9 % lleno | Completar en todas las operaciones |
| **Volumen (CBM)** | No se registra | Capturar (clave para LCL) |
| **Nº y tipo de contenedor** (20/40/HC/reefer) | 53 % lleno | Completar |
| **País de origen** | 52 % lleno | Completar |

### Prioridad 4 — Tarifarios (rate cards)
> Convierten ~25 % de los servicios de "predicho" a **lookup directo**.

| Concepto | Dato requerido |
|---|---|
| Descarga | Tarifa por contenedor del depósito/puerto, por año |
| Agenciamiento aduana | Tarifa fija por agente |
| Handling / THC | Tarifa naviera |
| SENASA / inspección | Tasas publicadas |

### Prioridad 5 — Distancia a destino y tipo de cambio por factura
> Reducen el error del transporte interno y el ruido cambiario.

| Dato | Formato |
|---|---|
| Distancia o zona del fundo de destino (km Callao → fundo) | km / zona |
| Tipo de cambio por factura (fecha real de cada gasto) | TC del día |
| Fechas completas del proceso (numeración, levante, retiro, vencimiento sobrestadía) | Fecha | → permite estimar la sobrestadía como `días de demora × tarifa` |

---

## 7. Qué habilita cada dato (impacto)

| Prioridad | Dato | Habilita | Impacto sobre el costo |
|---|---|---|---|
| 1 | HS + Ad Valorem | Tributos exactos | **65 % del total** |
| 2 | Monto de flete cotizado | Flete conocido | 36 % de servicios |
| 3 | Peso + contenedores + país | Costo variable real | Descarga + handling + aéreo |
| 4 | Tarifarios por proveedor × año | Servicios fijos = lookup | ~25 % de servicios |
| 5 | Distancia a fundo + TC por factura | Transporte T2 + ruido cambiario | 16 % de servicios + sesgo |

---

## 8. Cómo se usará: arquitectura objetivo (híbrida)

No se trata de un modelo "más grande", sino de **descomponer** el costo y atacar cada parte con la herramienta correcta:

```
COSTO TOTAL = Tributos        → fórmula (HS × CIF)          ~65 %  exacto
            + Flete           → monto cotizado del booking  ~13 %  conocido
            + Servicios fijos → tarifario (lookup)          ~15 %  tarifado
            + Variable (ML)   → peso, volumen, distancia,    ~7 %  predicho con
                                nº contenedores                     drivers reales
            + Sobrestadía     → banda P10–P90, no punto      ~3 %  estocástico
```

Cada bloque grande se vuelve casi-exacto y el Machine Learning solo carga con la parte **verdaderamente variable** (la más pequeña). Así el error agregado cae por debajo del 10 %.

El método de ML específico (gradient boosting, modelos por concepto, etc.) se elegirá según el dato disponible; **lo que limita hoy no es el método sino la información de entrada.**

---

## 9. Qué entregamos cuando los datos lleguen

| Entregable | Con datos de Prioridad 1–2 | Con datos de Prioridad 1–5 |
|---|---|---|
| Predicción de **costo total** por operación | Error ≤ 10 % (tributos + flete pasan a exactos) | — |
| Predicción de **servicios** por operación | Mejora sustancial | Error ≤ 10 % en el agregado y la mayoría de ops |
| Banda de incertidumbre P10–P90 | Sí | Sí (para operaciones volátiles) |
| Reentrenamiento periódico | Trimestral | Trimestral |

---

## 10. Lo que será siempre irreducible

La **sobrestadía / demurrage** (≈ 9 % de los servicios) depende de demoras portuarias, canal rojo de aforo e inspecciones: es **aleatoria**. Para esa porción nunca habrá un número exacto; se entrega un **intervalo P10–P90** de confianza, no un punto. Por eso la meta correcta es:

> **Predicción puntual ≤ 10 % en el agregado y en la mayoría de operaciones; banda de incertidumbre para las pocas operaciones volátiles.**

---

## 11. Resumen de la solicitud

| # | Pedimos | Para |
|---|---|---|
| 1 | **Partida HS + tasa Ad Valorem** por producto | Tributos exactos (65 % del total) |
| 2 | **Monto del flete cotizado** (no solo la fecha) | Flete conocido (36 % de servicios) |
| 3 | **Peso, volumen, nº/tipo de contenedor y país de origen** completos | Costo variable real |
| 4 | **Tarifarios por proveedor y año** | Servicios fijos = cálculo, no estimación |
| 5 | **Distancia al fundo, TC por factura y fechas completas del proceso** | Transporte interno, sobrestadía y ruido cambiario |

Con estos cinco bloques, el modelo deja de "adivinar" lo que en realidad son cálculos y tarifas, y la precisión esperada supera el 90 % (error ≤ 10 %) en el agregado y en la mayoría de las operaciones.

---

### Anexos técnicos (disponibles para Sistemas)

- Anatomía del costo: `data_csv/processed/target_por_concepto.parquet` (n = 1,058).
- Resultados de modelado y validación: `reports/modelado/14_modelado.md`, `reports/modelado/16_requisitos_90pct.md`.
- Evidencia de operaciones gemelas y disponibilidad de datos: scripts `src/ml/backtest_registradas.py` (modo honesto) y `src/ml/predecir_operacion.py`.
