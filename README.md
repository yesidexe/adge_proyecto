# Customer Churn Analytics & Predictive Modeling

Plataforma de análisis de datos a gran escala y modelado predictivo para la detección temprana de deserción de clientes (*churn*) en plataformas de comercio electrónico (*e-commerce*).

---

## 📌 Documentación del Proyecto

* 🏆 **[Informe.md](docs/Informe.md)**: Este documento desarrolla punto por punto las 12 secciones de la rúbrica (Tema, Sector, Objetivos, 5V del Big Data, Análisis Estadístico, Modelado Matemático, Resultados y Referencias).
* 🗺️ **[Plan.md](docs/Plan.md)**: Planificación técnica, diseño del stack tecnológico y justificación metodológica.
* 📋 **[Rubrica.md](docs/Rubrica.md)**: Guía y lineamientos de evaluación entregados.

---

## 🚀 Estructura del Repositorio

```text
Analisis_De_Datos/
├── data/
│   ├── raw/                 # Datos originales (ecommerce_data.csv y data_dictionary.csv)
│   └── processed/           # ecommerce_cleaned.csv + test_predictions.csv + customer_segments.csv
├── src/churn/               # Código Python reutilizable (datos, features, modelos, segmentación, predicción)
├── tests/                   # Pruebas automáticas (pytest, ~98 % de cobertura)
├── models/                  # Modelo de churn y segmentador entrenados (.joblib) + model_card.json
├── reports/                 # training_report.md, metrics.json y gráficos (figures/)
├── notebooks/
│   ├── 01_eda_limpieza.ipynb           # EDA, tratamiento de nulos, gráficos y correlaciones
│   ├── 02_modelado_predictivo.ipynb    # Feature Engineering, Regresión Logística vs XGBoost y ROC
│   └── 03_pipeline_reproducible.ipynb  # Entrenamiento paso a paso con el paquete src/churn + segmentación + predicción
├── docs/
│   ├── Informe.md        # Trabajo escrito formal sobre el proyecto
│   ├── Plan.md      # Planificación y arquitectura del proyecto
│   └── Rubrica.md # Rúbrica del proyecto
├── RESUMEN_EJECUTIVO.md                # Resumen del proyecto para negocio
├── pyproject.toml                      # Definición del paquete `churn` y configuración de pytest
├── requirements.txt                    # Dependencias esenciales
└── README.md                           # Presentación del proyecto
```

---

## 📊 Resultados de los Modelos en el Conjunto de Prueba ($N = 1.126$)

| Métrica | Regresión Logística (Baseline) | XGBoost Classifier (Ganador) |
| :--- | :---: | :---: |
| **Sensibilidad (Recall)** | 84.74% | **90.00%** |
| **Capacidad Discriminativa (ROC-AUC)** | 88.52% | **95.52%** |
| **Precisión** | 43.99% | **64.53%** |
| **F1-Score** | 57.91% | **75.16%** |
| **Exactitud (Accuracy)** | 79.22% | **89.96%** |

* **Matriz de Confusión en Test (XGBoost):**
  * **171** clientes en fuga detectados a tiempo ($TP$).
  * Solo **19** clientes en fuga no detectados ($FN$).
  * **842** clientes retenidos clasificados correctamente ($TN$).
  * **94** falsas alarmas ($FP$).

### Pipeline reproducible (`python -m churn.train`)

El paquete `src/churn/` repite el experimento con un `Pipeline` de scikit-learn: la imputación por mediana y el escalado se calculan **solo con el conjunto de entrenamiento** (sin *data leakage*), y se añade validación cruzada de 5 particiones. Mismo split (80/20, `random_state=42`):

| Métrica (prueba, N = 1.126) | Regresión Logística | XGBoost (seleccionado) |
| :--- | :---: | :---: |
| Recall | 84.74% | **90.53%** |
| ROC-AUC | 88.51% | **95.65%** |
| Precisión | 43.99% | **63.94%** |
| F1-Score | 57.91% | **74.95%** |

Las cifras difieren levemente de la tabla anterior (notebook 02) porque allí la mediana se calculó antes de dividir los datos. El detalle de cómo se entrenó el modelo está en [`reports/training_report.md`](reports/training_report.md) y en el notebook `03_pipeline_reproducible.ipynb`.

Además, el paquete **segmenta a los clientes** con K-Means (antigüedad, recencia, frecuencia y gasto; k elegido por silueta) y **predice** probabilidad de churn, nivel de riesgo (Bajo/Medio/Alto) y segmento para cualquier cliente:

```bash
python -m churn.train                                                   # entrena y regenera models/, reports/ y data/processed/
python -m churn.predict --input data/raw/ecommerce_data.csv --output predicciones.csv
pytest --cov=src --cov-report=term-missing                              # pruebas
```

---

## 💻 Instrucciones para ejecutar el código

### 1. Clonar el repositorio y acceder a la carpeta
```bash
git clone <URL_DEL_REPOSITORIO>
cd adge_proyecto
```

### 2. Crear el entorno virtual
Crea un entorno virtual aislado (nombrado `.venv`):
```bash
python3 -m venv .venv
```

### 3. Activar el entorno virtual
```bash
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\Activate.ps1         # Windows (PowerShell)
```

### 4. Instalar las dependencias
Con el entorno virtual ya activado, actualiza el gestor de paquetes e instala las librerías del proyecto (incluye Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn, JupyterLab e ipykernel):
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .          # instala el paquete `churn` (src/churn) para usar python -m churn.train
```

### 5. Iniciar Jupyter y ejecutar los notebooks
Inicia la interfaz web de **JupyterLab**:
```bash
jupyter lab
```
*(También puedes usar `jupyter notebook` o abrir el proyecto en VS Code y seleccionar como kernel el intérprete de `.venv`).*

Ejecuta los notebooks secuencialmente:
1. **`notebooks/01_eda_limpieza.ipynb`**: Análisis exploratorio, auditoría de calidad, imputación de nulos y exportación de datos limpios.
2. **`notebooks/02_modelado_predictivo.ipynb`**: Feature engineering, entrenamiento de Regresión Logística vs XGBoost y evaluación de métricas de negocio.
3. **`notebooks/03_pipeline_reproducible.ipynb`**: Mismo flujo con el paquete `src/churn` (pipeline sin fuga de datos, validación cruzada, segmentación y predicción con el modelo guardado).
