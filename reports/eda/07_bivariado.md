# 4.4 — Análisis bivariado: drivers del costo de servicios (n=1058)


## Costo de servicios por variable categórica (Kruskal-Wallis + ε²)

| variable        |   n_grupos |   kruskal_p |   epsilon2 | efecto    |
|:----------------|-----------:|------------:|-----------:|:----------|
| shipping_line   |         19 |     1.7e-80 |      0.473 | grande    |
| depot           |          9 |     2.5e-86 |      0.459 | grande    |
| type            |          6 |     2e-99   |      0.448 | grande    |
| categoria_canon |          8 |     2.5e-89 |      0.406 | grande    |
| mode            |          2 |     1.2e-79 |      0.342 | grande    |
| customs_agent   |          5 |     4.2e-66 |      0.302 | grande    |
| incoterm        |          7 |     3.4e-64 |      0.294 | grande    |
| incoterm_grupo  |          4 |     3.7e-54 |      0.236 | grande    |
| ffw             |         10 |     9.2e-40 |      0.212 | grande    |
| pais_origen     |          6 |     2.8e-11 |      0.109 | mediano   |
| canal           |          2 |     0.43    |     -0.001 | insignif. |

ε² (epsilon-cuadrado): >0.14 efecto grande, 0.06–0.14 mediano, <0.06 pequeño.

## Hipótesis: aéreo más caro por kg

- **AIR** (n=411): servicios mediana=$1257; costo/valor mediano=0.111 (servicios por USD de mercadería)
- **SEA** (n=633): servicios mediana=$6854; costo/valor mediano=0.159 (servicios por USD de mercadería)

## Costo vs variables de escala (Pearson/Spearman)

| variable     |    n |   pearson_log |   spearman |
|:-------------|-----:|--------------:|-----------:|
| bulks        | 1001 |         0.576 |      0.767 |
| amount_usd   | 1047 |         0.742 |      0.628 |
| peso_bruto   |   96 |         0.852 |      0.521 |
| qty          | 1041 |         0.391 |      0.435 |
| transit_days |  825 |         0.381 |      0.353 |
| ctnr_qty     |  564 |         0.053 |      0.026 |

## Matriz de correlación (Spearman) entre numéricas y target


**Multicolinealidad (|ρ|>0.6 entre predictores):** amount_usd~peso_bruto=0.64; amount_usd~target_servicios=0.63; amount_usd~tributos_usd=0.82; bulks~target_servicios=0.72; peso_bruto~transit_days=0.69