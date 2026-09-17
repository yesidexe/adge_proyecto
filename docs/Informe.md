# TRABAJO FINAL: ANALÍTICA DE DATOS A GRAN ESCALA
# Sistema Predictivo del Comportamiento del Usuario y Detección Temprana de Abandono (Churn) en Comercio Digital

**Asignatura:** Análisis de Datos a Gran Escala / Big Data  
**Modalidad:** Proyecto de Aplicación Práctica  
**Ponderación:** 40% del Semestre (Trabajo Escrito + Sustentación)  
**Fecha de Entrega:** Septiembre de 2026  

---

## Índice General de Contenidos

1. [Tema](#1-tema)
2. [Sector de Aplicación](#2-sector-de-aplicación)
3. [Objetivos del Proyecto (Taxonomía de Bloom)](#3-objetivos-del-proyecto-taxonomía-de-bloom)
   * 3.1 Objetivo General
   * 3.2 Objetivos Específicos
4. [Origen de los Datos](#4-origen-de-los-datos)
   * 4.1 Fuente y Características del Dataset
   * 4.2 Diccionario y Estructura de Variables
5. [Estrategia de Recolección de Datos](#5-estrategia-de-recolección-de-datos)
6. [Análisis de las 5V del Big Data](#6-análisis-de-las-5v-del-big-data)
7. [Análisis Estadístico](#7-análisis-estadístico)
   * 7.1 Análisis Univariado
   * 7.2 Análisis Bivariado y Correlaciones
8. [Fundamentación y Modelado Matemático](#8-fundamentación-y-modelado-matemático)
   * 8.1 Modelo Base: Regresión Logística Binaria
   * 8.2 Función de Costo y Optimización (Log-Loss Ponderada)
   * 8.3 Modelo Retador Avanzado: XGBoost Classifier
9. [Visualización de Datos](#9-visualización-de-datos)
10. [Análisis de Resultados y Métricas de Rendimiento](#10-análisis-de-resultados-y-métricas-de-rendimiento)
    * 10.1 Evaluación Comparativa en Conjunto de Prueba
    * 10.2 Análisis de la Matriz de Confusión
    * 10.3 Variables más Influyentes en el Abandono
11. [Conclusiones y Recomendaciones Operativas](#11-conclusiones-y-recomendaciones-operativas)
12. [Referencias Bibliográficas](#12-referencias-bibliográficas)

---

## 1. Tema

**Análisis predictivo del comportamiento del usuario y detección temprana de abandono (*customer churn*) en plataformas de comercio digital mediante técnicas de minería de datos y algoritmos de clasificación supervisada.**

El proyecto aborda la problemática de la fuga de clientes en entornos transaccionales de comercio electrónico, desarrollando un sistema capaz de identificar patrones de desafección y calcular la probabilidad individual de abandono antes de que el cliente cese definitivamente sus interacciones con la plataforma.

---

## 2. Sector de Aplicación

**Comercio Electrónico (*E-Commerce*) y Plataformas de Servicios Digitales.**

En este sector, la adquisición de nuevos compradores implica elevados costos de pauta digital y marketing (*Customer Acquisition Cost* - CAC). Estudios del sector demuestran que captar un cliente nuevo resulta entre **5 y 7 veces más costoso** que retener a un cliente actual. En consecuencia, maximizar el Valor del Tiempo de Vida del Cliente (*Customer Lifetime Value* - LTV) es la palanca principal de rentabilidad para cualquier negocio digital moderno.

---

## 3. Objetivos del Proyecto (Taxonomía de Bloom)

### 3.1 Objetivo General *(Nivel Bloom: Crear / Desarrollar)*
**Desarrollar** un modelo analítico predictivo de abandono de usuarios (**¿Qué?**), mediante técnicas de minería de datos, análisis exploratorio, imputación estadística y algoritmos de clasificación supervisada (**¿Cómo?**), con el fin de fundamentar estrategias proactivas y focalizadas de retención comercial en plataformas de comercio digital (**¿Para qué?**).

### 3.2 Objetivos Específicos ($OG = \sum OE$)

1. **Procesar** el conjunto de datos transaccionales brutos aplicando técnicas de estandarización categórica, tratamiento de valores faltantes mediante imputación por mediana y validación técnica, enmarcado en las dimensiones del Big Data. *(Nivel Bloom: Aplicar)*
2. **Analizar** el comportamiento del cliente mediante estadística descriptiva, tablas de contingencia y matrices de correlación (evaluando variables de antigüedad, recencia, quejas y gasto) para identificar los factores determinantes en la deserción. *(Nivel Bloom: Analizar)*
3. **Modelar** la probabilidad individual de deserción mediante el entrenamiento de un clasificador lineal paramétrico (Regresión Logística con ponderación de clases) y un clasificador no paramétrico de ensamble por gradiente (XGBoost). *(Nivel Bloom: Crear / Sintetizar)*
4. **Evaluar** el desempeño predictivo de los modelos utilizando métricas rigurosas (Sensibilidad/Recall, Exactitud, Precisión, F1-Score y área bajo la curva ROC-AUC), sintetizando los hallazgos en recomendaciones prácticas para la toma de decisiones empresariales. *(Nivel Bloom: Evaluar)*

---

## 4. Origen de los Datos

### 4.1 Fuente y Características del Dataset
* **Tipo:** Datos secundarios de acceso abierto.
* **Repositorio de Origen:** Kaggle Repository (*E-Commerce Customer Churn Analysis and Prediction*).
* **Volumen:** **5.630 perfiles de usuarios** con historial de navegación, transacciones y soporte técnico.
* **Variable Dependiente (Target):** `Churn` (Variable binaria: $1$ si el cliente abandonó el servicio en el último ciclo, $0$ si permanece activo).
* **Distribución de Clases:**
  * Clientes Retenidos (`0`): **4.682** registros (**83.16%**).
  * Clientes Desertores (`1`): **948** registros (**16.84%**).
  * Relación de desbalance moderado de aproximadamente $5:1$.

### 4.2 Diccionario y Estructura de Variables

| Variable | Tipo de Dato | Naturaleza | Descripción en el Negocio |
| :--- | :--- | :--- | :--- |
| `CustomerID` | Numérico (Entero) | Identificador | Clave única de auditoría del usuario (excluida del modelado). |
| `Churn` | Numérico (Binario) | Objetivo / Target | Estado del cliente: $1$ = Fuga / Deserción, $0$ = Retenido. |
| `Tenure` | Numérico (Continuo) | Temporal | Meses de permanencia activa del usuario en la plataforma. |
| `PreferredLoginDevice` | Texto (Nominal) | Comportamiento | Dispositivo de acceso preferente (`Mobile Phone`, `Computer`). |
| `CityTier` | Numérico (Ordinal) | Demográfico | Clasificación socioeconómica/urbanística de la ciudad (1, 2, 3). |
| `WarehouseToHome` | Numérico (Continuo) | Logístico | Distancia en kilómetros entre el almacén de despacho y el hogar. |
| `PreferredPaymentMode` | Texto (Nominal) | Transaccional | Método de pago habitual (`Debit Card`, `Credit Card`, `UPI`, etc.). |
| `Gender` | Texto (Binario) | Demográfico | Género registrado por el cliente (`Male`, `Female`). |
| `HourSpendOnApp` | Numérico (Continuo) | Comportamiento | Horas promedio diarias de navegación en la app móvil o sitio web. |
| `NumberOfDeviceRegistered`| Numérico (Discreto) | Comportamiento | Cantidad de dispositivos sincronizados a la cuenta. |
| `PreferedOrderCat` | Texto (Nominal) | Comercial | Categoría predominante de compra (`Mobile Phone`, `Grocery`, etc.). |
| `SatisfactionScore` | Numérico (Ordinal) | Percepción | Calificación subjetiva del servicio recibida por soporte (1 al 5). |
| `MaritalStatus` | Texto (Nominal) | Demográfico | Estado civil del cliente (`Single`, `Married`, `Divorced`). |
| `NumberOfAddress` | Numérico (Discreto) | Logístico | Total de direcciones de envío registradas por el usuario. |
| `Complain` | Numérico (Binario) | Soporte | Registro de quejas o reclamos en el último mes (1 = Sí, 0 = No). |
| `OrderAmountHikeFromlastYear`| Numérico (Continuo) | Financiero | Crecimiento porcentual del gasto respecto al año anterior. |
| `CouponUsed` | Numérico (Discreto) | Comercial | Total de cupones promocionales redimidos. |
| `OrderCount` | Numérico (Discreto) | Frecuencia | Total de órdenes de compra realizadas en la plataforma. |
| `DaySinceLastOrder` | Numérico (Continuo) | Recencia | Días transcurridos desde la última transacción (recencia). |
| `CashbackAmount` | Numérico (Continuo) | Fidelización | Promedio de reembolsos o cashback monetario acumulado. |

---

## 5. Estrategia de Recolección y Procesamiento

La estrategia de extracción y carga adoptó un flujo estructurado y reproducible:
1. **Extracción y Desacoplamiento:** Descarga secundaria desde el repositorio fuente y conversión programática de la hoja transaccional del archivo `.xlsx` a formato `.csv` delimitado, almacenado de forma inmutable en `data/raw/ecommerce_data.csv`.
2. **Ingesta Programática en Python:** Consumo mediante la librería `pandas` con tipado automático y verificación de integridad.
3. **Control de Calidad y Exportación:** Depuración de nulos, estandarización de categorías y exportación a la capa de datos procesados en `data/processed/ecommerce_cleaned.csv`.

---

## 6. Análisis de las 5V del Big Data

* **Volumen:** Se procesan 5.630 perfiles de clientes con 20 dimensiones (más de 112.000 puntos de datos transaccionales). El diseño modular del pipeline permite escalar a clústeres distribuidos y datasets de millones de filas sin alterar la lógica de preprocesamiento.
* **Velocidad:** En el comercio electrónico moderno, las transacciones, clics, quejas e inicios de sesión se producen de forma continua en tiempo real. La solución desacopla la fase de entrenamiento por lotes (*batch*) de la inferencia matemática rápida mediante vectores pre-escalados.
* **Variedad:** El conjunto integra variables cuantitativas continuas (`CashbackAmount`, `DaySinceLastOrder`), discretas (`OrderCount`), variables cualitativas nominales (`PaymentMode`, `Category`) y ordinales (`SatisfactionScore`, `CityTier`).
* **Veracidad:** Protocolos rigurosos de calidad de datos:
  * Eliminación de ambigüedades categóricas (ej. unificar `Phone` y `Mobile Phone`; `CC` y `Credit Card`).
  * Tratamiento de valores nulos mediante imputación por **mediana**, preservando la distribución y evitando el sesgo provocado por valores atípicos (*outliers*).
  * Verificación de duplicidad por `CustomerID` (0 duplicados detectados).
* **Valor:** Generación de un retorno económico directo para la empresa al predecir a tiempo el 90% de los clientes en riesgo de abandono, permitiendo activar promociones focalizadas y evitando la pérdida del LTV.

---

## 7. Análisis Estadístico

### 7.1 Análisis Univariado
Se calcularon las medidas de tendencia central y dispersión para las variables operativas:
* **Antigüedad (`Tenure`):** Mediana de 9 meses. Presenta asimetría positiva; los clientes con menos de 3 meses representan la mayor masa de observaciones y el mayor riesgo de abandono.
* **Recencia (`DaySinceLastOrder`):** Media de 4.5 días y mediana de 3 días. Existe una cola larga de usuarios inactivos que alcanza hasta 46 días sin compras.
* **Gasto y Cashback (`CashbackAmount`):** Promedio de \$177.2, con rango intercuartílico entre \$145 y \$196.

### 7.2 Análisis Bivariado y Factores de Riesgo
* **Impacto del Reclamo Técnico (`Complain` vs. `Churn`):**  
  Al cruzar la variable de quejas con el estado final del cliente, se encontró una de las relaciones estadísticas más contundentes:
  * Clientes sin quejas registradas (`Complain = 0`): Tasa de abandono del **10.8%**.
  * Clientes con quejas en el último mes (`Complain = 1`): Tasa de abandono del **31.7%**.  
  * *Hallazgo:* Presentar una queja formal casi **triplica la propensión al abandono**, convirtiendo a esta variable en un disparador crítico (*red flag*).
* **Relación Antigüedad vs. Churn:**  
  El diagrama de cajas (*boxplot*) evidencia que la mediana de antigüedad en clientes que desertaron es notablemente inferior (3 meses) a la de clientes retenidos (10 meses). Superado el primer año, la probabilidad de fuga decae a menos del 5%.
* **Matriz de Correlación Lineal:**  
  La correlación de Pearson confirma que `Tenure` posee la correlación negativa más fuerte con `Churn` ($r \approx -0.34$), mientras que `Complain` exhibe la correlación positiva más alta ($r \approx +0.25$).

---

## 8. Fundamentación y Modelado Matemático

### 8.1 Modelo Base: Regresión Logística Binaria Ponderada
La probabilidad condicional de que un cliente abandone la empresa ($Y = 1$) dado su vector de características normalizado $X \in \mathbb{R}^p$ se modela mediante la función logística estándar (sigmoide):
$$P(Y = 1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Donde el argumento $z$ es la combinación lineal ponderada de las $p = 25$ características resultantes del preprocesamiento:
$$z = \beta_0 + \sum_{j=1}^{p} \beta_j X_j = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p$$

* $\beta_0$: Término independiente (intercepto).
* $\beta_j$: Coeficiente de peso que determina la dirección y magnitud del impacto del predictor $X_j$. Un coeficiente positivo incrementa el logit de fuga ($\ln \frac{p}{1-p}$), mientras que uno negativo actúa como factor protector.

### 8.2 Función de Costo y Optimización (Entropía Cruzada Ponderada)
Para estimar los parámetros óptimos $\beta$, se minimiza la función de log-verosimilitud negativa (*Log-Loss*). Dado el desbalance de clases (83% vs 17%), una función de pérdida no balanceada tendería a ignorar a los desertores. Por ello, se introdujo una ponderación de clases inversamente proporcional a su frecuencia:
$$J(\beta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ w_1 \cdot y^{(i)} \ln(\hat{y}^{(i)}) + w_0 \cdot (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]$$

Donde los pesos se calculan como:
$$w_1 = \frac{N}{2 \cdot N_{\text{churn}}} \approx 2.97, \quad w_0 = \frac{N}{2 \cdot N_{\text{retenido}}} \approx 0.60$$
Esto penaliza con un costo ~5 veces mayor el error cuando el modelo predice erróneamente que un desertor permanecerá en la empresa.

### 8.3 Modelo Retador Avanzado: XGBoost Classifier
Para capturar fronteras de decisión no lineales y relaciones de interacción complejas (ej. cliente nuevo con queja no resuelta), se implementó un ensamble aditivo de árboles de decisión optimizados por gradiente:
$$\hat{y}_i^{(M)} = \sum_{m=1}^{M} f_m(x_i), \quad f_m \in \mathcal{F}$$

La optimización de XGBoost aproxima la pérdida mediante una expansión de Taylor de segundo orden:
$$\mathcal{L}^{(m)} \approx \sum_{i=1}^{N} \left[ g_i f_m(x_i) + \frac{1}{2} h_i f_m^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$
Donde $g_i$ y $h_i$ son el gradiente y el hessiano de la pérdida, $T$ es el número de hojas del árbol, y $\gamma, \lambda$ son términos de regularización estructural que previenen el sobreajuste (*overfitting*).

El balanceo de clases en XGBoost se controló formalmente mediante:
$$\text{scale\_pos\_weight} = \frac{N_{\text{retenido}}}{N_{\text{churn}}} = \frac{3.745}{759} \approx 4.94$$

---

## 9. Visualización de Datos

En el proyecto se generaron cuatro visualizaciones fundamentales para sustentar las decisiones:

1. **Matriz de Confusión Comparada (Heatmap de Seaborn):** Permite contrastar de manera intuitiva los aciertos y fallos de la Regresión Logística frente a XGBoost, evaluando directamente cuántos desertores fueron salvados y cuántos pasaron inadvertidos.
2. **Curva ROC Comparativa (Receiver Operating Characteristic):** Representa gráficamente la Tasa de Verdaderos Positivos frente a la Tasa de Falsos Positivos a través de todos los umbrales de decisión posibles, demostrando la superioridad global de XGBoost frente a la regresión y frente al azar ($AUC = 0.50$).
3. **Mapa de Calor de Correlaciones (Heatmap):** Visualización bidimensional con escala cromática divergente (`coolwarm`) de todas las correlaciones numéricas contra la variable objetivo `Churn`.
4. **Gráficos de Importancia de Variables (*Feature Importance* y Coeficientes $|\beta|$):** Gráficos de barras horizontales que clasifican los predictores con mayor ganancia de información en XGBoost y mayor magnitud en la Regresión Logística.

---

## 10. Análisis de Resultados y Métricas de Rendimiento

### 10.1 Evaluación Comparativa en el Conjunto de Prueba
La evaluación se efectuó estrictamente sobre los **1.126 clientes del conjunto de prueba independiente** (20% de los datos nunca observados durante el entrenamiento).

| Métrica de Rendimiento | Regresión Logística (Baseline) | XGBoost Classifier (Ganador) | Interpretación Operativa |
| :--- | :---: | :---: | :--- |
| **Exactitud (*Accuracy*)** | 79.22% | **89.96%** | Porcentaje de clasificaciones correctas sobre el total de la base. |
| **Sensibilidad (*Recall*)** | 84.74% | **90.00%** | **Capacidad de detectar a tiempo a los clientes que realmente desertaron.** |
| **Precisión (*Precision*)** | 43.99% | **64.53%** | Porcentaje de acierto cuando el modelo emite una alerta de fuga. |
| **F1-Score (Media Armónica)**| 57.91% | **75.16%** | Balance armónico global entre precisión y sensibilidad. |
| **Capacidad Discriminativa (*ROC-AUC*)**| 88.52% | **95.52%** | Capacidad del modelo de asignar mayor probabilidad de riesgo a un desertor. |

### 10.2 Análisis de la Matriz de Confusión (XGBoost en Test)
Sobre los 1.126 clientes del set de prueba:
* **Verdaderos Positivos ($TP = 171$):** Clientes que iban a desertar y fueron detectados exitosamente a tiempo para intervenir comercialmente.
* **Falsos Negativos ($FN = 19$):** Clientes que desertaron pero el modelo no logró predecir (solo el 10% de error de escape).
* **Verdaderos Negativos ($TN = 842$):** Clientes fieles clasificados correctamente como estables.
* **Falsos Positivos ($FP = 94$):** Clientes que no iban a desertar pero fueron catalogados como en riesgo (falsas alarmas con bajo costo de negocio).

### 10.3 Variables más Influyentes en la Deserción
El ranking de importancia obtenido por ganancia en XGBoost y magnitud de coeficientes en Regresión Logística señala los 5 factores dominantes:
1. **Antigüedad en la Plataforma (`Tenure` - Importancia: 18.18%):** Es el factor protector principal. Los usuarios novatos son los más susceptibles a abandonar tras una mala primera experiencia.
2. **Incidencia de Quejas (`Complain` - Importancia: 7.22%):** Factor acelerador de fuga número 1.
3. **Categoría de Compra (`Laptop & Accessory` - Importancia: 6.14%):** Clientes que adquieren tecnología de alto valor presentan dinámicas de retención diferenciadas frente a moda o consumo diario.
4. **Estado Civil (`MaritalStatus_Single` - Importancia: 5.40%):** Los clientes solteros presentan una volatilidad y propensión de rotación significativamente mayor que los clientes casados.
5. **Monto de Cashback (`CashbackAmount` - Importancia: 5.02%):** Los programas de recompensas y reembolsos actúan como un estabilizador clave de la fidelidad del cliente.

---

## 11. Conclusiones y Recomendaciones

### 11.1 Conclusiones Técnicas y Metodológicas
1. **Viabilidad y Superioridad Predictiva:** El modelo XGBoost Classifier demostró ser plenamente viable y altamente efectivo para operar como un sistema de alerta temprana, alcanzando una sensibilidad del **90.00%** y un **ROC-AUC del 95.52%**, superando ampliamente el baseline de la Regresión Logística (88.52%).
2. **Mitigación Exitosa del Desbalance:** La calibración del hiperparámetro `scale_pos_weight = 4.94` en XGBoost y `class_weight='balanced'` en Regresión Logística evitó que los algoritmos se sesgaran hacia la clase mayoritaria (83% retenidos), garantizando que solo se escaparan 19 de cada 190 clientes en fuga.
3. **Rigor contra la Fuga de Información:** La separación rigurosa de datos previa al escalado (`StandardScaler` ajustado únicamente en `train`) asegura que el modelo posee alta capacidad de generalización y no sufrirá degradación inesperada en un entorno de producción real.

### 11.2 Recomendaciones Operativas para el Negocio
1. **Protocolo Inmediato de Retención por Quejas (SLA < 24h):** Toda queja registrada por un usuario con menos de 6 meses de antigüedad debe detonar una alerta roja en el CRM de soporte, asignando un agente sénior con facultad para emitir compensaciones monetarias antes de que el usuario decida migrar a la competencia.
2. **Campañas Automatizadas por Recencia:** Establecer disparadores (*triggers*) de marketing automático cuando el indicador `DaySinceLastOrder` supere los 10 días respecto al ciclo habitual de compra del usuario, ofreciendo cupones dinámicos de envío gratuito o cashback en su categoría preferida.
3. **Segmentación Estratégica por Umbral de Riesgo:**
   * **Riesgo Crítico ($P(\text{churn}) \ge 0.70$):** Intervención de alto impacto (contacto personalizado, descuentos sustanciales).
   * **Riesgo Moderado ($0.40 \le P(\text{churn}) < 0.70$):** Notificaciones push de engagement, recomendaciones de catálogo y encuestas breves de satisfacción.
   * **Riesgo Bajo ($P(\text{churn}) < 0.40$):** Operación estándar sin desgaste en presupuesto de retención.

### 11.3 Limitaciones y Oportunidades de Escalamiento
* **Limitación:** El dataset actual representa una fotografía histórica agregada de comportamiento mensual.
* **Oportunidad:** Escalar el pipeline a un motor de procesamiento en streaming (como Apache Kafka o Spark Streaming) para capturar en tiempo real eventos granulares de sesión (ej. carritos abandonados consecutivamente o errores en la pasarela de pagos) y recalcular el score de riesgo de forma instantánea.

---

## 12. Referencias Bibliográficas (Formato IEEE / APA)

1. **Chen, T., & Guestrin, C.** (2016). *XGBoost: A Scalable Tree Boosting System*. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785
2. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer Series in Statistics. Springer New York.
3. **Verbeke, W., Martens, D., Mues, C., & Baesens, B.** (2012). *Building comprehensible customer churn prediction models with advanced rule induction techniques*. IEEE Transactions on Knowledge and Data Engineering, 24(12), 2100–2113. https://doi.org/10.1109/TKDE.2011.177
4. **Provost, F., & Fawcett, T.** (2013). *Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking*. O'Reilly Media.
5. **Geron, A.** (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
6. **Zhu, B., Baesens, B., & Backiel, A.** (2018). *Benchmarking state-of-the-art techniques for customer churn prediction in telecom and retail*. Decision Support Systems, 107, 79–89. https://doi.org/10.1016/j.dss.2018.01.006
7. **Verma, A.** (2021). *E-Commerce Customer Churn Analysis and Prediction Dataset*. Kaggle Repository. https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
