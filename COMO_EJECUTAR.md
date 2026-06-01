# Cómo ejecutar el proyecto y reproducir los mismos resultados

Guía paso a paso para levantar el entorno y regenerar **todos** los artefactos
(parquets procesados, informes EDA, modelos `.pkl`, métricas y figuras de modelado)
de forma idéntica a la versión publicada.

> **Regla de oro:** todos los comandos se ejecutan **desde la raíz del repositorio**
> (la carpeta que contiene `README.md`, `src/`, `data_csv/`). Varios scripts usan
> rutas relativas (`data_csv/raw`, `reports/eda/...`); si los corres desde otra
> carpeta, no encontrarán los datos.

---

## 1. Requisitos previos

| Herramienta | Versión usada | Notas |
|---|---|---|
| Python | **3.14.5** | Cualquier 3.11+ debería funcionar; los resultados publicados se generaron con 3.14. |
| Git | cualquiera | Para clonar el repo. |
| SO | Windows 11 (PowerShell) | También funciona en Linux/macOS (ver variantes bash). |
| Compilador C/C++ | sólo si `hdbscan` no trae wheel | En Windows: *Build Tools for Visual Studio*. En Linux: `build-essential`. |

Los datos crudos (`data_csv/raw/*.csv`, 8 archivos) **ya vienen versionados** en el repo,
así que no hay que descargarlos aparte.

---

## 2. Clonar y entrar al repo

```bash
git clone <URL-del-repo> ml-prediccion
cd ml-prediccion
```

---

## 3. Crear el entorno virtual e instalar dependencias

El proyecto asume un venv llamado `venv/` en la raíz (está en `.gitignore`).

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Si PowerShell bloquea la activación: `Set-ExecutionPolicy -Scope Process RemoteSigned`.

### Linux / macOS (bash)

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dependencias clave (ver `requirements.txt`): `pandas`, `numpy`, `scikit-learn`,
`lightgbm`, `xgboost`, `optuna`, `hdbscan`, `matplotlib`, `seaborn`, `pyarrow`.

---

## 4. Reproducibilidad (semilla)

La semilla global está fijada en [`src/lib/config.py`](src/lib/config.py): `SEED = 42`.
La usan el split temporal, los modelos (LightGBM/XGBoost/RandomForest) y el clustering.
Con la misma versión de librerías, los números (WAPE, MdAPE, importancias) salen iguales.

> Diferencias menores (último decimal) pueden aparecer si cambias la versión de
> `lightgbm`/`xgboost` o el número de hilos. Para máxima determinismo en multinúcleo
> puedes exportar `OMP_NUM_THREADS=1` antes de correr la fase ML.

---

## 5. Ejecutar todo el pipeline

Una sola orden regenera EDA + modelado de principio a fin:

```powershell
# Windows
venv\Scripts\python.exe src\run_all.py
```

```bash
# Linux/macOS (o con el venv activado, en cualquier SO)
python src/run_all.py
```

### Ejecutar sólo una fase

```bash
python src/run_all.py eda    # sólo EDA  (pasos 00..07)
python src/run_all.py ml     # sólo modelado (08, 09)
python src/run_all.py all    # ambas (= sin argumento)
```

Orden interno que ejecuta `run_all.py`:

| Fase | Scripts | Qué produce |
|---|---|---|
| EDA | `eda/00_inventario.py` → `eda/07_diagnostico.py` | `data_csv/processed/*.parquet`, `reports/eda/*.md`, `reports/eda/figures/*.png` |
| ML | `ml/08_modelado.py`, `ml/09_figuras.py` | `models/*.pkl`, `reports/modelado/*.md`, `*.csv`, `figures/*.png` |

---

## 6. Ejecutar scripts individuales (opcional)

Todos desde la raíz del repo:

```bash
python src/ml/08_modelado.py     # entrena, valida (CV temporal), predice pendientes, guarda modelos
python src/ml/09_figuras.py      # figuras del reporte de modelado
python src/ml/_audit.py          # auditoría técnica: leakage / disponibilidad / nulos
```

> `src/lib/` es **librería**, no se ejecuta sola. Contiene `config.py` (rutas, semilla,
> mapa de columnas, política del target), `utils.py`, `dataset.py` y `features.py`.
> Los scripts de `eda/` y `ml/` añaden `src/lib` al `sys.path` e importan por nombre.

---

## 7. Probar el modelo entrenado

Requiere haber corrido antes la fase ML (paso 5 o `08_modelado.py`), que deja los
`.pkl` en `models/`.

```bash
# Predicción manual de UNA operación (costo de servicios sin tributos, con intervalo P10–P90)
python src/ml/predecir_operacion.py              # interactivo, pregunta campo por campo
python src/ml/predecir_operacion.py --ejemplo    # ejemplo precargado
python src/ml/predecir_operacion.py --json mi_operacion.json   # desde JSON (dict o lista)

# Backtest sobre operaciones YA registradas (mide error real vs costo real)
python src/ml/backtest_registradas.py                 # in-sample (rápido, optimista)
python src/ml/backtest_registradas.py --modo honesto  # holdout temporal (cifra de despliegue)
```

---

## 8. Regenerar el notebook de presentación

```bash
python src/_build_notebook.py    # reescribe notebooks/EDA_presentacion.ipynb
```

El notebook **lee resultados ya generados** (no reentrena), así que primero corre el
pipeline del paso 5.

---

## 9. Dónde quedan los resultados

```
data_csv/processed/   *.parquet          # dataset modelable + target (generados por EDA)
models/               *.pkl, model_meta.json   # artefactos entrenados (gitignored)
reports/eda/          00..13_*.md + figures/   # informes y figuras del EDA
reports/modelado/     14..16_*.md, *.csv, figures/   # reporte ML, métricas, predicciones
```

`models/` está en `.gitignore` (binarios grandes): no se versiona, se **regenera**
corriendo la fase ML.

---

## 10. Verificar que reprodujiste los mismos resultados

Tras `python src/run_all.py ml`, compara con `reports/modelado/metrics_global.csv`.
Valores de referencia (validación temporal 2025, despliegue):

| modelo | WAPE | MdAPE |
|---|---:|---:|
| baseline (mediana × segmento) | 0.50 | 0.45 |
| **LightGBM** | 0.37 | 0.28 |
| XGBoost | 0.35 | 0.23 |

Intervalos P10–P90 conformal (CQR): cobertura ≈ 0.76. Detalle completo en
[`reports/modelado/14_modelado.md`](reports/modelado/14_modelado.md).

---

## 11. Problemas frecuentes

| Síntoma | Causa / solución |
|---|---|
| `FileNotFoundError: data_csv/raw/...` | No estás en la raíz del repo. Haz `cd` a la carpeta del proyecto. |
| `ModuleNotFoundError: config` | Ejecuta vía `run_all.py` o el script directo de `eda/`/`ml/` (ellos arman el `sys.path`); no muevas `src/lib/`. |
| Falla al instalar `hdbscan` | Falta compilador C++. Instala *Build Tools for Visual Studio* (Windows) o `build-essential` (Linux), luego reinstala. |
| `predecir_operacion.py` no encuentra modelo | Corre primero la fase ML (`python src/run_all.py ml`) para generar `models/*.pkl`. |
| Números levemente distintos | Versiones distintas de `lightgbm`/`xgboost` o hilos. Fija `requirements.txt` y prueba `OMP_NUM_THREADS=1`. |
