# 4.3 — Análisis univariado (target fiable, n=1058)


## Distribución del target — servicios logísticos (USD)

- **servicios**: media=7556, mediana=4790, p5=83, p95=26364, skew=8.72, curtosis=135.34, CV=1.63
- **tributos**: media=14054, mediana=6430, p5=0, p95=63470, skew=4.43, curtosis=28.51, CV=1.79
- **total**: media=21610, mediana=13142, p5=194, p95=79486, skew=3.74, curtosis=22.57, CV=1.36

Log-normalidad (servicios): Shapiro p=1.2e-29 crudo vs p=8.2e-14 en log.
- Outliers severos (>Q3+3·IQR = >31924): 33 ops (3.2%), máx=241365. No se eliminan (operaciones caras legítimas).

## Variables numéricas

|                |   count |    mean |     50% |      95% |            max |   %nulos |
|:---------------|--------:|--------:|--------:|---------:|---------------:|---------:|
| amount_usd     |    1058 | 82554.3 | 34346.6 | 356425   |     3.4827e+06 |      0   |
| qty            |    1058 | 88469.6 |   155   | 605880   |     3.168e+06  |      0   |
| bulks          |    1058 |  1159.2 |    13.5 |   1405.8 |     1.0119e+06 |      0   |
| ctnr_qty       |     566 |     1.1 |     1   |      1   |    11          |     46.5 |
| peso_bruto     |      96 | 13221.1 | 11965.5 |  24689.5 | 26760          |     90.9 |
| payment_term   |    1010 |    38.9 |    30   |     90   |   270          |      4.5 |
| transit_days   |    1047 |    23.2 |    11   |     81   |   152          |      1   |
| n_oc_distintas |    1058 |     1.2 |     1   |      2   |     7          |      0   |
| n_productos    |    1058 |     1.2 |     1   |      2   |     5          |      0   |

## Variables categóricas — cardinalidad y categorías raras

| variable        |   cardinalidad |   pct_nulos | top                  |   cat_raras_n |
|:----------------|---------------:|------------:|:---------------------|--------------:|
| supplier        |            143 |         0   | PROJAR (106)         |           124 |
| pol             |             83 |         0.4 | SANTIAGO (157)       |            70 |
| shipping_line   |             62 |         2.8 | CMA CGM (143)        |            43 |
| ffw             |             35 |         2.9 | DIRECTO (461)        |            25 |
| punto_llegada   |             32 |         6.2 | FUNDO ARMONIA (193)  |            19 |
| pod             |             28 |         0.1 | CALLAO (569)         |            23 |
| depot           |             28 |         6.1 | IMUPESA (445)        |            19 |
| buyer           |             24 |         0   | RENE (235)           |            11 |
| pais_origen     |             23 |        48.1 | ESPAÑA (184)         |            17 |
| customs_agent   |             14 |         0.9 | AVM (697)            |             9 |
| incoterm        |             10 |         0.1 | EXW (365)            |             3 |
| categoria_canon |              9 |         0   | SUSTRATOS (281)      |             1 |
| type            |              7 |         0.1 | AIR (411)            |             1 |
| incoterm_grupo  |              4 |         0.1 | vendedor_flete (478) |             0 |
| mode            |              3 |         0.2 | SEA (633)            |             1 |

- **Alta cardinalidad** (supplier, pol, producto): requieren target/frequency encoding, no one-hot.
- **Categorías raras** (n<15): agrupar en 'OTROS' o usar suavizado bayesiano para evitar sobreajuste.