# Guía de Proyecto: Analítica de Datos y Big Data

## Información General y Evaluación
* **Modalidad:** Tema Libre.
* **Ponderación:** 40% del corte/semestre (Trabajo escrito + Presentación).
* **Fecha y hora límite de entrega:** 17/09/2026 hasta las 6:30 p.m. (Vía correo).

---

## 1. Tema
Análisis predictivo del comportamiento del usuario y detección temprana de abandono (*churn*) en plataformas de comercio digital.

## 2. Sector
Comercio electrónico (*E-commerce* y servicios digitales).

## 3. Objetivos (Estructurados con Taxonomía de Bloom)

### Objetivo General (Nivel Bloom: Crear / Desarrollar)
Desarrollar un modelo analítico predictivo de abandono de usuarios (**¿Qué?**), mediante técnicas de minería de datos, análisis exploratorio y algoritmos de clasificación supervisada (**¿Cómo?**), con el fin de fundamentar estrategias proactivas de retención comercial en plataformas de comercio digital (**¿Para qué?**).

### Objetivos Específicos ($OG = \sum OE$)
1. **Procesar** el conjunto de datos transaccionales brutos aplicando técnicas de limpieza, estructuración y validación técnica, bajo el marco de las 5V del Big Data. *(Nivel Bloom: Aplicar)*
2. **Analizar** el comportamiento del cliente mediante estadística descriptiva y matrices de correlación (variables de frecuencia, recencia y gasto), para identificar los patrones determinantes en la deserción. *(Nivel Bloom: Analizar)*
3. **Modelar** la probabilidad de abandono individual a través del entrenamiento y ajuste de algoritmos de clasificación supervisada (como Regresión Logística o Árboles de Decisión). *(Nivel Bloom: Crear / Sintetizar)*
4. **Evaluar** el rendimiento predictivo del modelo mediante métricas de clasificación (precisión, sensibilidad, F1-score y matriz de confusión), sintetizando los hallazgos en visualizaciones de datos para la toma de decisiones. *(Nivel Bloom: Evaluar)*

---

## 4. Origen de los Datos
* **Tipo:** Externo (repositorio público de datos abiertos).
* **Fuente recomendada:** Kaggle / UCI Machine Learning Repository.
  * *Datasets sugeridos:*
    * *E-Commerce Customer Behavior and Churn Dataset* (Kaggle).
    * *Brazilian E-Commerce Public Dataset by Olist* (Kaggle).
    * *Telco Customer Churn Dataset* (adaptable a comercio digital por suscripción).
* **Estructura típica de variables:**
  * `CustomerID`: Identificador único.
  * `Tenure` / `Antiguedad`: Meses de permanencia en la plataforma.
  * `Recency`: Días transcurridos desde la última interacción o compra.
  * `OrderCount`: Número total de transacciones realizadas.
  * `CashbackAmount` / `Monetary`: Gasto total o volumen transaccionado.
  * `Complain`: Registro binario de quejas o incidencias con soporte.
  * `Churn`: Variable objetivo binaria (1 = Deserción, 0 = Activo).

## 5. Estrategia de Recolección de Datos
* Extracción secundaria estructurada mediante descarga y carga directa de archivos delimitados (`.csv`).
* Consumo programático o ingesta mediante scripts de análisis en Python (utilizando librerías como Pandas o Polars).

---

## 6. Análisis de las 5V del Big Data
* **Volumen:** Magnitud de los datos (cantidad de filas/registros y peso en megabytes del dataset seleccionado).
* **Velocidad:** Naturaleza dinámica del e-commerce, donde los eventos de navegación, clics y transacciones se generan en tiempo real de forma continua.
* **Variedad:** Coexistencia de variables numéricas discretas/continuas, atributos categóricos, fechas y texto estructurado.
* **Veracidad:** Protocolos de calidad de datos aplicados (imputación o eliminación de valores nulos, detección de *outliers* o registros atípicos, y eliminación de duplicados).
* **Valor:** Impacto en el negocio derivado de la reducción del costo de adquisición de clientes (CAC) al evitar fugas de ingresos mediante ofertas personalizadas.

---

## 7. Análisis Estadístico
* **Univariado:** Cálculo de medidas de tendencia central (media, mediana) y de dispersión (desviación estándar, rangos intercuartílicos), junto con histogramas de distribución para gasto, recencia y compras.
* **Bivariado:**
  * Coeficientes de correlación (Pearson o Spearman) entre variables continuas y el estado de abandono.
  * Tablas de contingencia y pruebas de chi-cuadrado para variables categóricas (ej. quejas vs. churn).

---

## 8. Modelo Matemático
* **Algoritmo principal:** Regresión Logística Binaria.
* **Formulación matemática:**
  La probabilidad $P(Y=1)$ de que un usuario abandone la plataforma está dada por la función sigmoide:
  $$P(Y=1 | X) = \frac{1}{1 + e^{-z}}$$
  Donde:
  $$z = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n$$
  * $\beta_0$: Intercepto.
  * $\beta_i$: Coeficientes ponderados que representan el peso e impacto de cada variable predictora $X_i$ (recencia, quejas, gasto, etc.).
* **Función de optimización (Loss Function):** Entropía cruzada binaria (Log-Loss):
  $$J(\beta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

---

## 9. Visualización de Datos
* **Matriz de Confusión:** Visualización de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos.
* **Curva ROC-AUC:** Evaluación de la tasa de verdaderos positivos frente a falsos positivos a diferentes umbrales.
* **Mapa de Calor (Heatmap):** Matriz de correlación entre todas las variables numéricas.
* **Importancia de Variables (Feature Importance):** Gráfico de barras ordenando los factores más críticos que empujan al cliente al abandono.

---

## 10. Análisis de Resultados
* Desempeño métrico: Precisión (*Precision*), Sensibilidad (*Recall*), *F1-Score* y Exactitud (*Accuracy*).
* Identificación de las variables más influyentes: Demostración de qué factor (por ejemplo, incremento en días de inactividad o quejas sin resolver) eleva exponencialmente el riesgo de deserción.

---

## 11. Conclusiones
* Viabilidad técnica del modelo para operar como sistema de alerta temprana.
* Recomendaciones directas para el equipo de marketing y soporte al cliente (intervención proactiva antes de que el cliente cumpla el umbral de fuga).
* Limitaciones del conjunto de datos y oportunidades de escalamiento a streaming en tiempo real.

---

## 12. Referencias Bibliográficas
* Documentación oficial y literatura académica en formato APA / IEEE sobre minería de datos, modelos de predicción de abandono (*churn prediction*) y marcos teóricos de Big Data.
