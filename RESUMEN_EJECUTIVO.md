# Resumen Ejecutivo — Customer Churn Analytics

## ¿Qué hace el proyecto?

Predice **qué clientes de una tienda en línea están a punto de abandonarla (churn)** para que el negocio actúe antes de perderlos. A partir del historial de cada cliente (antigüedad, días sin comprar, quejas, satisfacción, cashback, método de pago, etc.), un modelo estima su probabilidad de fuga y permite priorizar acciones de retención.

## El problema de negocio

Captar un cliente nuevo cuesta entre 5 y 7 veces más que retener uno existente. Cada cliente que se va se lleva ingresos recurrentes y el valor de vida (*LTV*) ya invertido en captarlo. Sin una señal temprana, la empresa descubre la fuga cuando ya es tarde.

## Datos

- **Fuente:** dataset público *E-Commerce Customer Churn* (Kaggle).
- **Tamaño:** ~5.630 clientes × 20 variables; la variable objetivo `Churn` indica si el cliente abandonó (1) o sigue activo (0).
- **Preparación:** se unifican categorías inconsistentes (p. ej. `Phone`/`Mobile Phone`, `COD`/`Cash on Delivery`) y los valores faltantes se imputan con la mediana, dejando un dataset limpio sin nulos.

## Enfoque

1. **Análisis exploratorio:** qué factores se asocian al abandono (quejas, antigüedad, recencia, satisfacción) y cómo se correlacionan.
2. **Modelado:** se comparan dos algoritmos con el mismo split 80 % entrenamiento / 20 % prueba:
   - *Regresión Logística* — modelo base, fácil de explicar.
   - *XGBoost* — modelo avanzado, captura relaciones no lineales.
3. **Evaluación orientada al negocio:** se prioriza el **Recall** (detectar a quien de verdad se va), porque perder un cliente sin avisar cuesta más que enviar una promoción de más.

## Resultados (conjunto de prueba, 1.126 clientes)

| Métrica | Regresión Logística | **XGBoost** |
| :--- | :---: | :---: |
| Recall (clientes en fuga detectados) | 84,7 % | **90,0 %** |
| ROC-AUC | 88,5 % | **95,5 %** |
| Precisión | 44,0 % | **64,5 %** |
| F1-Score | 57,9 % | **75,2 %** |

XGBoost detecta **171 de 190** clientes en fuga (solo 19 se escapan) con 94 falsas alarmas, y supera las metas del informe (Recall ≥ 85 %, AUC ≥ 88 %, F1 ≥ 70 %, Precisión ≥ 55 %).

## Recomendaciones para el negocio

1. **Reforzar a los clientes recientes:** la antigüedad protege; concentrar el acompañamiento en los primeros meses.
2. **Atender las quejas con prioridad:** los reclamos son uno de los principales aceleradores de la fuga.
3. **Vigilar la recencia:** un aumento de los días desde la última compra es una señal temprana de inactividad.

## Estado actual y siguientes pasos

- **Hoy:** análisis completo en tres notebooks (`notebooks/`), informe formal en `docs/Informe.md` y `Informe.pdf`, y un **paquete Python reproducible** (`src/churn/`, con pruebas automáticas) que entrena los modelos, guarda el modelo (`models/`), genera el reporte de entrenamiento (`reports/training_report.md`) y **predice el riesgo de abandono y el segmento de cada cliente** (`python -m churn.predict`).
- **Segmentación:** además de predecir el abandono, los clientes se agrupan en 3 segmentos por comportamiento (antigüedad, recencia, frecuencia y gasto), con tasas de churn real muy distintas (de ~4 % a ~22 %), útiles para diseñar campañas diferenciadas.
- **Nota sobre las cifras:** el pipeline reproducible calcula la imputación solo con datos de entrenamiento; por eso XGBoost queda en Recall 90,5 % y ROC-AUC 95,7 % (vs 90,0 % y 95,5 % de la tabla de arriba).
- **Pendiente:** es un piloto local; no hay aplicación ni API para que marketing y soporte consulten las predicciones. El modelo debe validarse con datos reales antes de usarse en producción.
- **Escalamiento:** la metodología es aplicable a la base completa de clientes de una organización.

## Cómo explorarlo

| Qué | Dónde |
| :--- | :--- |
| Presentación y ejecución del código | [`README.md`](README.md) |
| Informe formal (12 secciones de la rúbrica) | [`docs/Informe.md`](docs/Informe.md) · [`Informe.pdf`](Informe.pdf) |
| Plan técnico y justificación del stack | [`docs/Plan.md`](docs/Plan.md) |
| Cómo se entrenó el modelo (datos, pipeline, métricas, gráficos) | [`reports/training_report.md`](reports/training_report.md) |
| Código Python y pruebas | [`src/churn/`](src/churn/) · [`tests/`](tests/) |
| Modelo entrenado | [`models/`](models/) |
| Notebooks (EDA, modelado y pipeline reproducible) | [`notebooks/`](notebooks/) |
| Datos original y limpio | [`data/raw/`](data/raw/) · [`data/processed/`](data/processed/) |
