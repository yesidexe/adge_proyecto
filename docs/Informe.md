# Sistema Predictivo del Comportamiento del Usuario y Detección Temprana de Abandono (Churn) en Comercio Digital

## Índice

1. [Tema](#1-tema)
2. [Sector](#2-sector)
3. [Objetivos](#3-objetivos)
   * 3.1 [Objetivo General](#31-objetivo-general-nivel-bloom-crear--desarrollar)
   * 3.2 [Objetivos Específicos](#32-objetivos-específicos)
4. [Origen de los datos](#4-origen-de-los-datos)
   * 4.1 [Dataset de Prueba y Justificación](#41-dataset-de-prueba-y-justificación)
   * 4.2 [Estructura y Diccionario de Variables](#42-estructura-y-diccionario-de-variables)
5. [Estrategia de recolección de datos](#5-estrategia-de-recolección-de-datos)
   * 5.1 [Flujo de Procesamiento y Capas](#51-flujo-de-procesamiento-y-capas)
   * 5.2 [Herramientas de Trabajo](#52-herramientas-de-trabajo)
6. [Análisis de las 5V del Big Data](#6-análisis-de-las-5v-del-big-data)
7. [Análisis estadístico](#7-análisis-estadístico)
   * 7.1 [Análisis Univariado](#71-análisis-univariado)
   * 7.2 [Análisis Bivariado y Correlaciones](#72-análisis-bivariado-y-correlaciones)
8. [Modelo matemático](#8-modelo-matemático)
   * 8.1 [Regresión Logística Binaria (Modelo Base)](#81-regresión-logística-binaria-modelo-base)
   * 8.2 [Función de Pérdida con Balanceo de Clases](#82-función-de-pérdida-con-balanceo-de-clases)
   * 8.3 [XGBoost Classifier (Modelo Avanzado)](#83-xgboost-classifier-modelo-avanzado)
9. [Visualización de datos](#9-visualización-de-datos)
10. [Análisis de resultados](#10-análisis-de-resultados)
    * 10.1 [Estrategia de Validación (Train/Test)](#101-estrategia-de-validación-traintest)
    * 10.2 [Métricas de Rendimiento Clave](#102-métricas-de-rendimiento-clave)
    * 10.3 [Criterio de Evaluación en la Matriz de Confusión](#103-criterio-de-evaluación-en-la-matriz-de-confusión)
11. [Conclusiones](#11-conclusiones)
    * 11.1 [Viabilidad Técnica y Aporte Metodológico](#111-viabilidad-técnica-y-aporte-metodológico)
    * 11.2 [Recomendaciones para la Toma de Decisiones](#112-recomendaciones-para-la-toma-de-decisiones)
    * 11.3 [Escalamiento del Análisis](#113-escalamiento-del-análisis)
12. [Referencias bibliográficas](#12-referencias-bibliográficas)

---

## 1. Tema

**Análisis predictivo del comportamiento del usuario y detección temprana de abandono (*customer churn*) en plataformas de comercio digital mediante minería de datos y algoritmos de clasificación supervisada.**

El proyecto aborda la problemática de la fuga de clientes en tiendas en línea. A través del análisis de hábitos de compra, tiempos de entrega, reclamos e inactividad, se plantea diseñar un modelo predictivo capaz de estimar la probabilidad de que un usuario abandone la plataforma, identificando a tiempo los factores de riesgo antes de que deje de comprar.

---

## 2. Sector

**Comercio Electrónico (*E-Commerce*) y Servicios Digitales.**

En el sector del comercio digital, captar a un cliente nuevo a través de publicidad suele costar entre **5 y 7 veces más** que mantener a uno existente (*Customer Acquisition Cost* - CAC frente al *Customer Lifetime Value* - LTV).

Cuando un comprador habitual se marcha, la tienda pierde ingresos recurrentes. Por esta razón, contar con un sistema que identifique qué clientes están en riesgo permite enfocar las acciones comerciales y de soporte en los usuarios que realmente lo necesitan, protegiendo la rentabilidad de la tienda.

---

## 3. Objetivos

### 3.1 Objetivo General
**Desarrollar** un modelo analítico predictivo para la detección de abandono de usuarios (**¿Qué?**), aplicando técnicas de minería de datos, análisis exploratorio, imputación estadística y algoritmos de clasificación supervisada (**¿Cómo?**), con el fin de fundamentar estrategias comerciales de retención proactivas y focalizadas (**¿Para qué?**).

### 3.2 Objetivos Específicos ($OG = \sum OE$)

1. **Procesar** el conjunto de datos transaccionales mediante limpieza de inconsistencias de texto, tratamiento de nulos por mediana y validación de duplicados, bajo el marco conceptual de las 5V del Big Data. 
2. **Analizar** el comportamiento del cliente mediante estadística descriptiva y matrices de correlación (evaluando antigüedad, recencia, quejas y gasto) para identificar los factores determinantes en la deserción. 
3. **Modelar** la probabilidad individual de deserción comparando el entrenamiento de un clasificador lineal (Regresión Logística con balanceo de clases) y un ensamble de árboles por gradiente (XGBoost).
4. **Evaluar** el rendimiento del modelo mediante métricas de clasificación (priorizando la Sensibilidad o *Recall*, junto con ROC-AUC y F1-Score), traduciendo los hallazgos en recomendaciones para la toma de decisiones.

---

## 4. Origen de los datos

### 4.1 Dataset de Prueba y Justificación
* **Fuente:** Repositorio público abierto de Kaggle (*E-Commerce Customer Churn Analysis and Prediction* - Ankit Verma).
* **Naturaleza de la muestra:** Para la fase de diseño, formulación matemática y validación metodológica se utiliza una **muestra de prueba de aproximadamente 5.600 clientes**.
* **Justificación técnica:** Este conjunto funciona como un entorno piloto controlado. Permite diseñar el pipeline de preprocesamiento y comparar los algoritmos de forma ágil y reproducible en un equipo local, con la ventaja de que el flujo queda estructurado para escalarse a bases de datos corporativas más grandes.
* **Premisa de desbalance:** En el comercio electrónico, los clientes que se van representan naturalmente una proporción menor frente a los que se quedan activos. La metodología asume este desbalance como punto de partida para configurar algoritmos con ajuste de pesos.

### 4.2 Estructura y Diccionario de Variables
La gran ventaja de esta muestra es su **riqueza dimensional**: reúne **20 variables** que abarcan logística, soporte, finanzas, comportamiento y percepción:

| Variable | Tipo de Dato | Dimensión | Rol en el Negocio |
| :--- | :--- | :--- | :--- |
| `CustomerID` | Entero | Auditoría | Identificador único del cliente (se excluye del modelado). |
| `Churn` | Binario (0/1) | **Objetivo** | Estado del usuario: $1$ = Abandonó la plataforma, $0$ = Sigue activo. |
| `Tenure` | Numérico | Antigüedad | Meses de permanencia activa del cliente en la tienda. |
| `PreferredLoginDevice` | Texto | Canal | Dispositivo principal de acceso (`Mobile Phone`, `Computer`). |
| `CityTier` | Ordinal (1, 2, 3) | Demográfico | Nivel socioeconómico o tamaño de la ciudad de residencia. |
| `WarehouseToHome` | Numérico | **Logístico** | Distancia en kilómetros entre el almacén y el domicilio del cliente. |
| `PreferredPaymentMode` | Texto | Transaccional | Método de pago habitual (`Credit Card`, `Debit Card`, `Cash on Delivery`, `E-wallet`). |
| `Gender` | Texto | Demográfico | Género registrado (`Male`, `Female`). |
| `HourSpendOnApp` | Numérico | Hábitos | Horas promedio diarias dedicadas a navegar en la app o web. |
| `NumberOfDeviceRegistered`| Entero | Hábitos | Cantidad de dispositivos vinculados a la cuenta del usuario. |
| `PreferedOrderCat` | Texto | Comercial | Categoría en la que más compra (`Laptop & Accessory`, `Mobile Phone`, `Fashion`, `Grocery`). |
| `SatisfactionScore` | Entero (1 al 5) | **Percepción** | Calificación de satisfacción asignada por el usuario en encuestas. |
| `MaritalStatus` | Texto | Demográfico | Estado civil registrado (`Single`, `Married`, `Divorced`). |
| `NumberOfAddress` | Entero | Logístico | Total de direcciones de entrega registradas en su perfil. |
| `Complain` | Binario (0/1) | **Soporte** | Si el usuario radicó una queja en el último mes ($1$ = Sí, $0$ = No). |
| `OrderAmountHikeFromlastYear`| Numérico | Financiero | Crecimiento porcentual del gasto respecto al año anterior. |
| `CouponUsed` | Entero | Promocional | Total de cupones de descuento redimidos. |
| `OrderCount` | Entero | Frecuencia | Número total de órdenes de compra realizadas en la plataforma. |
| `DaySinceLastOrder` | Numérico | **Recencia** | Días transcurridos desde su última compra confirmada. |
| `CashbackAmount` | Numérico | Fidelización | Promedio monetario de reembolsos o cashback recibido. |

---

## 5. Estrategia de recolección de datos

### 5.1 Flujo de Procesamiento y Capas
Para asegurar el orden y la reproducibilidad, el flujo de datos se organiza en capas separadas:

```text
data/raw/ecommerce_data.csv          <- Archivo original inmutable (solo lectura)
          │
          ▼  (Limpieza léxica + Imputación por mediana + One-Hot Encoding)
data/processed/ecommerce_cleaned.csv  <- Dataset limpio con 0 nulos listo para modelar
          │
          ▼
notebooks/ (01_eda_limpieza.ipynb y 02_modelado_predictivo.ipynb)
```

1. **Capa cruda (`data/raw/`):** Almacena el archivo CSV descargado de la fuente. Permanece intacto como punto de respaldo y auditoría.
2. **Capa procesada (`data/processed/`):** Guarda el dataset resultante tras la estandarización de categorías de texto (ej. unificar `Phone` con `Mobile Phone`) y la imputación de nulos mediante la **mediana**, preservando la estabilidad estadística ante valores extremos.
3. **Aislamiento contra fuga de información (*Data Leakage*):** Las operaciones de escalado numérico (`StandardScaler`) se calculan únicamente sobre los datos de entrenamiento y se aplican después sobre el conjunto de prueba.

### 5.2 Herramientas de Trabajo
* **Lenguaje:** Python.
* **Entorno interactivo:** Cuadernos de **JupyterLab** para la ejecución estructurada (`notebooks/01_eda_limpieza.ipynb` y `notebooks/02_modelado_predictivo.ipynb`).
* **Librerías principales:** `pandas` y `numpy` para manipulación de tablas, `matplotlib` y `seaborn` para visualización gráfica, y `scikit-learn` junto con `xgboost` para el modelado analítico.

---

## 6. Análisis de las 5V del Big Data

* **Volumen:** El proyecto utiliza una muestra analítica de prueba de aproximadamente 5.600 clientes con 20 variables de negocio (más de 112.000 datos en total). Este volumen permite desarrollar y probar el flujo de trabajo de forma rápida en un entorno local, sin necesidad de infraestructura costosa. Al mismo tiempo, el modelado se plantea mediante matrices vectorizadas, lo que permite que esta misma metodología sea la base para aplicarse sobre volúmenes masivos de datos empresariales.
* **Velocidad:** En el comercio electrónico las compras, quejas y clics ocurren continuamente. El sistema plantea un procesamiento por lotes periódico (ej. evaluaciones periódicas del riesgo de clientes), permitiendo calcular la probabilidad de abandono de forma ágil para la toma de decisiones.
* **Variedad:** El conjunto reúne tipos de datos muy diversos: variables cuantitativas continuas (distancias, dinero de cashback), conteos discretos (órdenes, dispositivos), variables cualitativas nominales (métodos de pago, categorías) y escalas de percepción (estrellas de satisfacción).
* **Veracidad:** La calidad del dato es esencial para que el modelo sea confiable. El plan contempla auditar duplicados, estandarizar textos y tratar valores nulos mediante la mediana para no introducir sesgos artificiales.
* **Valor:** Es el beneficio directo para el negocio. Al anticipar con precisión qué usuarios tienen alta probabilidad de abandonar la tienda, la empresa puede focalizar sus recursos en retenerlos antes de que sea tarde.

---

## 7. Análisis estadístico

El análisis estadístico se divide en dos fases que responden directamente a lo estructurado en los cuadernos de trabajo:

### 7.1 Análisis Univariado
Examina el comportamiento individual de cada variable del dataset:
* **Medidas de tendencia central y dispersión:** Cálculo de media, mediana, desviación estándar y rango intercuartílico ($IQR$) para variables numéricas clave como antigüedad (`Tenure`), recencia (`DaySinceLastOrder`) y gasto (`CashbackAmount`).
* **Análisis de forma:** Detección de asimetrías y presencia de clientes atípicos (por ejemplo, clientes con tiempos prolongados de inactividad).
* **Distribución de variables categóricas:** Proporciones de clientes según su método de pago preferido, dispositivo de acceso y categoría de producto habitual.

### 7.2 Análisis Bivariado y Correlaciones
Estudia cómo interactúan las variables explicativas con la variable de abandono (`Churn`):
* **Correlación lineal de Pearson:** Evalúa el grado de asociación lineal entre las variables continuas y el estado de abandono, esperando validar una correlación negativa entre la antigüedad (`Tenure`) y la deserción.
* **Impacto de reclamos (`Complain` vs `Churn`):** Comparación de la tasa de abandono entre clientes con quejas registradas frente a clientes sin reclamos, contrastando si las quejas actúan como un disparador del abandono.
* **Diagramas de cajas (*Boxplots*):** Comparación visual de la distribución de días desde la última compra (`DaySinceLastOrder`) entre clientes retenidos y clientes que desertaron.

---

## 8. Modelo matemático

Se formulan dos modelos con propósitos complementarios: uno lineal para máxima interpretabilidad y otro de ensamble para capturar relaciones complejas.

### 8.1 Regresión Logística Binaria (Modelo Base)
Modela la probabilidad condicional de que un cliente abandone la tienda ($P(Y=1)$) dado su vector de características $X$, utilizando la función sigmoide:

$$P(Y = 1 \mid X) = \frac{1}{1 + e^{-z}}$$

Donde $z$ representa la combinación lineal de las características del cliente ponderadas por sus coeficientes $\beta$:

$$z = \beta_0 + \sum_{j=1}^{p} \beta_j X_j = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p$$

* $\beta_0$: Intercepto del modelo.
* $\beta_j$: Coeficiente asociado a cada variable. Un valor positivo aumenta el riesgo de abandono; un valor negativo actúa como factor protector de permanencia.
* **Interpretabilidad:** Permite calcular el cambio en las razones de momios (*Odds Ratio* = $e^{\beta_j}$), facilitando explicar el impacto de cada factor al equipo de negocio.

### 8.2 Función de Pérdida con Balanceo de Clases
Dado que los clientes que abandonan son una minoría, una función de error estándar tendería a predecir a favor de la mayoría. Para corregir este sesgo, se implementa la **Entropía Cruzada Binaria Ponderada (Log-Loss con pesos)**:

$$J(\beta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ w_1 \cdot y^{(i)} \ln(\hat{y}^{(i)}) + w_0 \cdot (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]$$

Donde los pesos $w_1$ y $w_0$ se calculan en función de la frecuencia inversa de cada clase, penalizando con mayor costo cuando el modelo comete el error de no detectar a un cliente que se marcha.

### 8.3 XGBoost Classifier (Modelo Avanzado)
Para capturar relaciones no lineales e interacciones múltiples (por ejemplo, clientes novatos con reclamos pendientes), se formula un ensamble aditivo de árboles de decisión por gradiente (**XGBoost**).

La predicción final es la suma ponderada de $M$ árboles de decisión:

$$\hat{y}_i^{(M)} = \sum_{m=1}^{M} f_m(x_i)$$

La función objetivo en cada etapa incluye una aproximación de Taylor de segundo orden y un término explícito de regularización para evitar el sobreajuste (*overfitting*):

$$\mathcal{L}^{(m)} \approx \sum_{i=1}^{N} \left[ g_i f_m(x_i) + \frac{1}{2} h_i f_m^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$

Donde $g_i$ y $h_i$ son el gradiente y el hessiano de la pérdida, $T$ es el número de hojas terminales del árbol, y $\gamma, \lambda$ son parámetros que controlan la complejidad del modelo.

---

## 9. Visualización de datos

Se diseñan cuatro visualizaciones analíticas para sustentar el diagnóstico y la toma de decisiones:

1. **Matriz de Confusión Comparada (Heatmap):** Permite contrastar los aciertos y fallos de la Regresión Logística frente a XGBoost, identificando directamente cuántos desertores fueron detectados y cuántos pasaron inadvertidos.
2. **Curva ROC-AUC Comparada:** Evalúa la Tasa de Verdaderos Positivos frente a Falsos Positivos a lo largo de distintos umbrales de corte, midiendo la capacidad discriminativa global de los modelos frente al azar ($AUC = 0.50$).
3. **Mapa de Calor de Correlaciones (Heatmap):** Cuadrícula visual con escala de color para inspeccionar rápidamente la fuerza y dirección de las asociaciones entre las variables numéricas y el abandono.
4. **Gráfico de Importancia de Variables (*Feature Importance*):** Diagrama de barras horizontales ordenado de mayor a menor que señala con claridad cuáles son las variables más determinantes en la decisión del modelo (ej. antigüedad, quejas, recencia).

---

## 10. Análisis de resultados

### 10.1 Estrategia de Validación (Train/Test)
* **Partición Estratificada:** Los datos se dividirán en un **80% para entrenamiento** y un **20% reservado para prueba independiente**.
* **Conservación del balance:** Se conservará la proporción natural de desertores en ambas particiones para evitar sesgos muestrales.
* **Evaluación ciega:** Las transformaciones de escala se ajustarán únicamente en los datos de entrenamiento para asegurar que el conjunto de prueba sea una evaluación rigurosa.

### 10.2 Métricas de Rendimiento Clave
En problemas de abandono, guiarse únicamente por la Exactitud (*Accuracy*) resulta engañoso debido al desbalance natural. Por ello, la evaluación se centra en:

| Métrica | Propósito en el Negocio | Criterio Esperado |
| :--- | :--- | :---: |
| **Sensibilidad (*Recall*)** | **Métrica principal.** Mide qué porcentaje de los clientes que verdaderamente se iban fueron detectados a tiempo. Minimiza los clientes perdidos sin detección. | **$\ge 85.0\%$** |
| **ROC-AUC** | Capacidad general del modelo para ordenar a los usuarios asignando mayor riesgo a quienes realmente abandonarán. | **$\ge 88.0\%$** |
| **F1-Score** | Media armónica entre la precisión y la sensibilidad, garantizando un equilibrio operativo adecuado. | **$\ge 70.0\%$** |
| **Precisión** | Proporción de alertas correctas generadas por el modelo, evitando desgastar esfuerzos en falsas alarmas. | $\ge 55.0\%$ |

### 10.3 Criterio de Evaluación en la Matriz de Confusión
Las decisiones analíticas se orientan bajo una perspectiva de costos asimétricos:
* **Falso Negativo (cliente en fuga que no se detecta):** Representa el mayor costo para el negocio, pues se pierde el valor futuro del cliente y la inversión realizada en captarlo.
* **Falso Positivo (cliente que no se iba pero recibe una alerta):** Tiene un impacto menor; solo representa el costo de enviarle un beneficio o comunicación comercial.
* *Criterio:* Se prioriza la **Sensibilidad** sobre la precisión pura, buscando reducir al mínimo los clientes desertores inadvertidos.

---

## 11. Conclusiones

### 11.1 Viabilidad Técnica y Aporte Metodológico
1. **Factibilidad del modelado predictivo:** La propuesta metodológica confirma que es técnicamente viable anticipar el abandono de clientes en comercio digital combinando un modelo lineal explicativo (Regresión Logística) con un ensamble no lineal robusto (XGBoost).
2. **Tratamiento del desbalance:** La incorporación de funciones de pérdida ponderadas permite abordar el desbalance de clases inherente al comercio electrónico sin sesgar las predicciones hacia la clase mayoritaria.
3. **Aislamiento experimental:** El pipeline de preprocesamiento estructurado por capas garantiza una evaluación rigurosa y libre de contaminación (*data leakage*).

### 11.2 Recomendaciones para la Toma de Decisiones
A partir del análisis del comportamiento del cliente, se derivan tres lineamientos estratégicos:
1. **Foco en clientes de reciente ingreso:** La antigüedad suele actuar como un factor protector clave; por ello, los esfuerzos de acompañamiento y fidelización deben concentrarse en los primeros meses del comprador.
2. **Atención prioritaria a quejas:** Las incidencias con soporte representan uno de los principales aceleradores de la fuga. Se recomienda que los clientes que registren reclamos reciban seguimiento prioritario antes de que decidan abandonar la tienda.
3. **Gestión analítica de la recencia:** Monitorear el incremento anormal en los días transcurridos desde la última compra para identificar oportunamente a los clientes que comienzan a mostrar signos de inactividad.

### 11.3 Escalamiento del Análisis
* **Muestra de prueba como base:** Los datos de prueba de aproximadamente 5.600 clientes sirve para validar la metodología y afinar los algoritmos localmente.
* **Proyección futura:** La formulación matricial y el diseño modular del código permiten que este mismo marco metodológico pueda aplicarse posteriormente sobre la base total de clientes o conectarse a las fuentes de datos completas de la organización.

---

## 12. Referencias bibliográficas

1. **Chen, T., & Guestrin, C.** (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. https://doi.org/10.1145/2939672.2939785
2. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer Series in Statistics. https://doi.org/10.1007/978-0-387-84858-7
3. **Provost, F., & Fawcett, T.** (2013). *Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking*. O'Reilly Media.
4. **Verbeke, W., Martens, D., Mues, C., & Baesens, B.** (2012). *Building comprehensible customer churn prediction models with advanced rule induction techniques*. IEEE Transactions on Knowledge and Data Engineering, 24(12), 2100–2113. https://doi.org/10.1109/TKDE.2011.177
5. **Géron, A.** (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
6. **Verma, A.** (2021). *E-Commerce Customer Churn Analysis and Prediction Dataset*. Kaggle Repository. https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
