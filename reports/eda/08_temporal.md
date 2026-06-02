# 4.5 — Tiempos del proceso, estacionalidad y derivadas (n=1058)


## Tiempos del proceso (días)

|               |   count |   mean |   50% |   90% |   max |
|:--------------|--------:|-------:|------:|------:|------:|
| transit_days  |    1047 |   23.2 |    11 |    63 |   152 |
| dias_num_dam  |     238 |    6   |     2 |    12 |   362 |
| dias_levante  |     702 |    3.9 |     1 |    10 |   365 |
| dias_deposito |     583 |    9.1 |     6 |    17 |   121 |
| dias_a_fundo  |     861 |   14.7 |    12 |    27 |   369 |

## Relación días en depósito ↔ sobrestadía/almacenaje

- Spearman(días depósito, costo sobrestadía) = **0.53** (n=583)
- Operaciones con sobrestadía>0: 154 (días depósito mediana con sobrestadía=13 vs sin=5)

## Tendencia temporal del costo (inflación de tarifas)

|   anio |   n |   serv_mediana |   unit_valor_med |   flete_med |
|-------:|----:|---------------:|-----------------:|------------:|
|   2020 | 127 |        2270.45 |            0.066 |           0 |
|   2021 | 261 |        5721.46 |            0.141 |           0 |
|   2022 | 128 |        5194.41 |            0.091 |         119 |
|   2023 | 147 |        2127.2  |            0.15  |          57 |
|   2024 | 218 |        5838.4  |            0.19  |           0 |
|   2025 | 176 |        5954.19 |            0.195 |         390 |
|   2026 |   1 |          33.73 |            6.487 |           0 |

- Tendencia log(costo/valor) ~ año: pendiente=+0.140/año (≈+15.0%/año), p=2.0e-07. Hay deriva temporal significativa ⇒ **validación temporal (forward chaining), no K-fold aleatorio.**

## Variables derivadas propuestas (con justificación de negocio)

| variable          | fórmula                      | justificación                                                  |
|:------------------|:-----------------------------|:---------------------------------------------------------------|
| transit_days      | ata − atd                    | Tránsito largo (marítimo) ⇒ más flete y riesgo de sobrestadía. |
| dias_deposito     | retiro_t1 − descarga         | Días en depósito temporal ⇒ almacenaje/sobrestadía.            |
| es_aereo          | mode == AIR                  | Separa la estructura de costo aérea vs marítima.               |
| incoterm_grupo    | incoterm → quién paga flete  | EXW/FOB: importador asume flete (más costo logístico).         |
| categoria_canon   | categoría armonizada         | Colapsa la deriva semántica de CATEGORY a ~8 grupos.           |
| costo_unit_valor  | servicios / valor_mercadería | Ratio más estable; normaliza por tamaño de la operación.       |
| ratio_flete_valor | flete / valor                | Intensidad logística relativa al valor.                        |
| canal_rojo        | canal == rojo                | Aforo físico ⇒ más inspección/tiempo/costo.                    |
| requiere_senasa   | tiene fecha SENASA           | Plantas/sustratos ⇒ trámite fitosanitario extra.               |
| tiene_seguro      | insurance_hper == SI         | Operaciones aseguradas (perfil de riesgo/valor).               |
| mes_arribo / anio | de ata                       | Estacionalidad e inflación de tarifas.                         |
| zona_ruta         | POL→POD agrupado             | Corredor logístico (Chile-Callao, Europa-Callao, USA-aéreo).   |