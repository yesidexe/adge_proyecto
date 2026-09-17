# Customer Churn Analytics & Predictive Modeling

Plataforma de análisis de datos a gran escala y modelado predictivo para la detección temprana de deserción de clientes (*churn*) en plataformas de comercio electrónico (*e-commerce*).

---

## 📌 Documentación del Proyecto

* 🏆 **[docs/Informe_final.md](docs/Informe_final.md)**: **Documento formal de entrega (40% de la nota)**. Desarrolla punto por punto las 12 secciones de la rúbrica del docente (Tema, Sector, Objetivos Bloom, 5V del Big Data, Análisis Estadístico, Modelado Matemático con fórmulas LaTeX, Resultados y Referencias APA/IEEE).
* 🗺️ **[docs/Plan_de_trabajo.md](docs/Plan_de_trabajo.md)**: Planificación técnica, diseño del stack tecnológico y justificación metodológica.
* 📋 **[docs/Rubrica_del_proyecto.md](docs/Rubrica_del_proyecto.md)**: Guía y lineamientos de evaluación entregados por el docente.

---

## 🚀 Estructura del Repositorio

```text
adge_proyecto/
├── data/
│   ├── raw/                 # Datos originales (ecommerce_data.csv y data_dictionary.csv)
│   └── processed/           # Dataset limpio con 0 nulos (ecommerce_cleaned.csv)
├── notebooks/
│   ├── 01_eda_limpieza.ipynb           # EDA, tratamiento de nulos, gráficos y correlaciones
│   └── 02_modelado_predictivo.ipynb    # Feature Engineering, Regresión Logística vs XGBoost y ROC
├── docs/
│   ├── Informe_final.md        # Trabajo escrito formal para entrega
│   ├── Plan_de_trabajo.md      # Planificación y arquitectura del proyecto
│   └── Rubrica_del_proyecto.md # Rúbrica y guía del docente
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

---

## 💻 Instrucciones para Reproducir el Proyecto

1. Activar el entorno virtual:
   ```bash
   source .venv/bin/activate
   ```
2. Abrir y ejecutar los notebooks secuencialmente:
   * **`notebooks/01_eda_limpieza.ipynb`**
   * **`notebooks/02_modelado_predictivo.ipynb`**
