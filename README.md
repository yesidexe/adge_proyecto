# Customer Churn Analytics & Prediction

Proyecto de analítica y modelado predictivo para la detección temprana de deserción de clientes (*churn*) en plataformas de comercio electrónico (*e-commerce*).

---

## 📌 Documentación Principal

El plan de trabajo detallado, la justificación de datos, la estructura y el paso a paso se encuentran en:
👉 **[docs/GUIA.md](file:///workspaces/adge_proyecto/docs/GUIA.md)**

---

## 🚀 Estructura del Repositorio

```text
adge_proyecto/
├── data/
│   ├── raw/                 # Archivo de datos original (Kaggle)
│   └── processed/           # Dataset limpio listo para modelado
├── notebooks/
│   ├── 01_eda_limpieza.ipynb           # Análisis exploratorio, nulos y correlaciones
│   └── 02_modelado_predictivo.ipynb    # Regresión Logística, XGBoost y métricas
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Funciones para carga de datos
│   └── features.py          # Preprocesamiento y escalado
├── docs/
│   └── GUIA.md              # Roadmap y especificación del proyecto
├── requirements.txt         # Librerías esenciales del entorno
└── README.md                # Presentación del proyecto
```

---

## 📦 Librerías Clave
* **Procesamiento:** `pandas`, `numpy`
* **Visualización:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`, `xgboost`