# 4.6 — Análisis multivariado y de segmentos (n=1058)


## Segmentos: dónde se concentra el gasto de servicios

**Mediana de servicios (USD) por categoría × modo:**
| categoria_canon   |   AIR |   LAND |   SEA |
|:------------------|------:|-------:|------:|
| AGROQUIMICOS      | 37578 |    nan |  4789 |
| EMPAQUE           |  1428 |   8377 |  7837 |
| EQUIPOS_INSTR     |   738 |    nan |  3394 |
| MAQUINARIA        |   799 |    nan |  5646 |
| MUESTRAS          |    44 |    nan |   nan |
| OTROS             |   117 |    nan |   nan |
| PLANTAS           |  3520 |    nan |   nan |
| REPUESTOS         |  1012 |    nan |  3925 |
| SUSTRATOS         |    90 |   2915 |  8989 |

**Nº de operaciones por categoría × modo:**
| categoria_canon   |   AIR |   LAND |   SEA |
|:------------------|------:|-------:|------:|
| AGROQUIMICOS      |     3 |      0 |   139 |
| EMPAQUE           |    61 |     11 |   181 |
| EQUIPOS_INSTR     |    61 |      0 |    12 |
| MAQUINARIA        |     4 |      0 |    26 |
| MUESTRAS          |    15 |      0 |     0 |
| OTROS             |     1 |      0 |     0 |
| PLANTAS           |   123 |      0 |     0 |
| REPUESTOS         |   134 |      0 |     5 |
| SUSTRATOS         |     9 |      1 |   270 |

## Gasto total de servicios por segmento (top 10)

|                          |   n |            total |   mediana |   %_del_total |
|:-------------------------|----:|-----------------:|----------:|--------------:|
| ('SUSTRATOS', 'SEA')     | 270 |      3.71357e+06 |      8989 |            46 |
| ('EMPAQUE', 'SEA')       | 181 |      1.72594e+06 |      7837 |            22 |
| ('AGROQUIMICOS', 'SEA')  | 139 | 690517           |      4789 |             9 |
| ('EMPAQUE', 'AIR')       |  61 | 588103           |      1428 |             7 |
| ('PLANTAS', 'AIR')       | 123 | 566178           |      3520 |             7 |
| ('MAQUINARIA', 'SEA')    |  26 | 193334           |      5646 |             2 |
| ('REPUESTOS', 'AIR')     | 134 | 138507           |      1012 |             2 |
| ('EMPAQUE', 'LAND')      |  11 | 113161           |      8377 |             1 |
| ('AGROQUIMICOS', 'AIR')  |   3 | 108194           |     37578 |             1 |
| ('EQUIPOS_INSTR', 'SEA') |  12 |  65157           |      3394 |             1 |

## Mediana de servicios por país × grupo de incoterm

| pais_origen    |   comprador_flete |   comprador_todo |   vendedor_destino |   vendedor_flete |
|:---------------|------------------:|-----------------:|-------------------:|-----------------:|
| ALEMANIA       |               646 |              269 |                nan |              nan |
| BELGICA        |               nan |              nan |                255 |              nan |
| BRASIL         |               nan |              nan |                129 |              nan |
| CANADA         |               nan |             6967 |                201 |              nan |
| CHILE          |              9222 |             3995 |                  0 |             3942 |
| CHINA          |             12016 |              294 |                 31 |            10790 |
| COLOMBIA       |               nan |              nan |                 21 |              nan |
| ECUADOR        |               nan |              nan |               2915 |              nan |
| ESPAÑA         |              1672 |             5365 |                 25 |            12568 |
| ESTADOS UNIDOS |               nan |              nan |               1801 |             2149 |
| ESTONIA        |              1525 |              nan |                191 |            10878 |
| INDIA          |               nan |              nan |                 48 |              nan |
| ISRAEL         |               nan |              437 |                 32 |              796 |
| ITALIA         |               nan |              nan |                nan |             5389 |
| JAPON          |               nan |             1516 |                nan |              nan |
| LETONIA        |               nan |              nan |                nan |             8254 |
| MARRUECOS      |               nan |              nan |                nan |             3312 |
| MEXICO         |               nan |             2708 |                  0 |             5000 |
| NUEVA ZELANDA  |               nan |             1271 |                nan |              nan |
| PAISES BAJOS   |               nan |             4121 |                 31 |             9385 |
| POLONIA        |               nan |              nan |                  0 |              nan |
| SRI LANKA      |               nan |              nan |                 91 |             9030 |
| USA            |               948 |             1027 |                nan |             1725 |

## Clustering no supervisado (KMeans) — perfiles de operación

- k óptimo por silueta = **4** (silhouette=0.44)
|   cluster |   n |   serv_mediana |   valor_med |   bulks_med |   pct_aereo | categoria_top   |
|----------:|----:|---------------:|------------:|------------:|------------:|:----------------|
|         0 | 102 |             71 |          30 |           1 |          97 | EMPAQUE         |
|         1 | 616 |           7077 |       50570 |          60 |           0 | SUSTRATOS       |
|         2 | 328 |           1686 |       13014 |           2 |          95 | PLANTAS         |
|         3 |  12 |           7099 |       23232 |          52 |           0 | SUSTRATOS       |

## Importancia preliminar de variables (LightGBM, split temporal)

- Split: train (≤2023, n=663) → test (≥2024, n=395).
- **MAE test = $3,468** | **MdAPE = 24.9%** | **WAPE = 41.8%** (baseline exploratorio, sin tuning ni features derivadas completas).

**Importancia (ganancia y permutación sobre test):**
| feature         |   gain_% |   perm |
|:----------------|---------:|-------:|
| bulks           |     13.4 |  0.384 |
| amount_usd      |     22.1 |  0.225 |
| categoria_canon |      1.7 |  0.168 |
| shipping_line   |      2.5 |  0.168 |
| incoterm_grupo  |      1.8 |  0.124 |
| ctnr_qty        |      2.5 |  0.112 |
| depot           |      1   |  0.043 |
| transit_days    |     14.7 |  0.042 |
| mes_arribo      |     10   |  0.017 |
| mode            |      0.9 |  0.017 |
| payment_term    |      7.2 |  0.009 |
| qty             |     16.5 |  0.006 |
| type            |      0.1 |  0.002 |
| n_productos     |      0.4 |  0     |
| n_oc_distintas  |      0.1 |  0     |
| anio            |      3.5 |  0     |
| pais_origen     |      0   |  0     |
| customs_agent   |      0.7 | -0.003 |