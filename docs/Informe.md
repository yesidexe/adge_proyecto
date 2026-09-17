# INFORME DE PROPUESTA TÉCNICA
# Sistema Predictivo del Comportamiento del Usuario y Detección Temprana de Abandono (Churn) en Comercio Digital

## Índice General de Contenidos

1. [Tema](#1-tema)
2. [Sector de Aplicación](#2-sector-de-aplicación)
3. [Objetivos del Proyecto](#3-objetivos-del-proyecto-taxonomía-de-bloom)
   * 3.1 [Objetivo General](#31-objetivo-general-nivel-bloom-crear--desarrollar)
   * 3.2 [Objetivos Específicos](#32-objetivos-específicos)
4. [Origen de los Datos](#4-origen-de-los-datos)
   * 4.1 [Fuente y Justificación del Dataset](#41-fuente-y-justificación-del-dataset)
   * 4.2 [Estructura y Diccionario de Variables](#42-estructura-y-diccionario-de-variables)
5. [Estrategia de Recolección y Pipeline de Procesamiento](#5-estrategia-de-recolección-y-pipeline-de-procesamiento)
   * 5.1 [Flujo de Ingesta y Desacoplamiento](#51-flujo-de-ingesta-y-desacoplamiento)
   * 5.2 [Tecnologías y Entorno de Desarrollo](#52-tecnologías-y-entorno-de-desarrollo)
6. [Análisis de las 5V del Big Data](#6-análisis-de-las-5v-del-big-data)
7. [Plan de Análisis Estadístico](#7-plan-de-análisis-estadístico)
   * 7.1 [Análisis Univariado Proyectado](#71-análisis-univariado-proyectado)
   * 7.2 [Análisis Bivariado y Contraste de Relaciones](#72-análisis-bivariado-y-contraste-de-relaciones)
8. [Fundamentación y Modelado Matemático](#8-fundamentación-y-modelado-matemático)
   * 8.1 [Modelo Paramétrico Base: Regresión Logística Binaria](#81-modelo-paramétrico-base-regresión-logística-binaria)
   * 8.2 [Función de Pérdida y Tratamiento del Desbalance (Log-Loss Ponderada)](#82-función-de-pérdida-y-tratamiento-del-desbalance-log-loss-ponderada)
   * 8.3 [Modelo Retador de Ensamble: XGBoost Classifier](#83-modelo-retador-de-ensamble-xgboost-classifier)
9. [Propuesta de Visualización de Datos](#9-propuesta-de-visualización-de-datos)
10. [Metodología de Evaluación y Resultados Esperados](#10-metodología-de-evaluación-y-resultados-esperados)
    * 10.1 [Estrategia de Partición y Validación](#101-estrategia-de-partición-y-validación)
    * 10.2 [Métricas Clave y Criterios de Éxito del Negocio](#102-métricas-clave-y-criterios-de-éxito-del-negocio)
    * 10.3 [Protocolo de Matriz de Confusión y Análisis de Costos](#103-protocolo-de-matriz-de-confusión-y-análisis-de-costos)
11. [Conclusiones Esperadas e Impacto Operativo](#11-conclusiones-esperadas-e-impacto-operativo)
    * 11.1 [Impacto Esperado en Marketing y Retención](#111-impacto-esperado-en-marketing-y-retención)
    * 11.2 [Limitaciones Identificadas y Hoja de Ruta de Escalamiento](#112-limitaciones-identificadas-y-hoja-de-ruta-de-escalamiento)
12. [Referencias Bibliográficas](#12-referencias-bibliográficas)

---

## 1. Tema

**Análisis predictivo del comportamiento del usuario y detección temprana de abandono (*customer churn*) en plataformas de comercio digital mediante técnicas de minería de datos y algoritmos de clasificación supervisada.**

El proyecto propone una solución integral de analítica predictiva para abordar la desafección de clientes en plataformas de comercio electrónico (*e-commerce*). A través de la modelación del comportamiento histórico del usuario (interacciones, quejas, recencia y frecuencia transaccional), se diseñará un sistema capaz de estimar la probabilidad individual de fuga de cada usuario antes de que cese sus compras, transformando datos transaccionales brutos en alertas comerciales tempranas.

---

## 2. Sector de Aplicación

**Comercio Electrónico (*E-Commerce*) y Plataformas Transaccionales de Servicios Digitales.**

En el ecosistema del comercio electrónico, el Costo de Adquisición de Clientes (*Customer Acquisition Cost* - CAC) ha aumentado significativamente debido a la saturación de los canales publicitarios digitales. Diversos estudios de la industria confirman que adquirir un nuevo comprador cuesta entre **5 y 7 veces más** que fidelizar y retener a un cliente preexistente. 

La pérdida imprevista de clientes (*churn*) deteriora de forma directa el Valor del Tiempo de Vida del Cliente (*Customer Lifetime Value* - LTV) y reduce la rentabilidad operativa. En este contexto, anticipar qué usuarios planean abandonar la tienda virtual permite desplegar campañas de rescate automatizadas y personalizadas, optimizando el presupuesto de mercadeo y salvaguardando el flujo de ingresos de la empresa.

---

## 3. Objetivos del Proyecto

### 3.1 Objetivo General *(Nivel Bloom: Crear / Desarrollar)*
**Desarrollar** un modelo analítico predictivo de abandono de usuarios, mediante la formulación de técnicas de minería de datos, análisis exploratorio univariado/bivariado, protocolos de imputación estadística y algoritmos de clasificación supervisada, con el fin de fundamentar estrategias proactivas y focalizadas de retención comercial en plataformas de comercio digital.

### 3.2 Objetivos Específicos ($OG = \sum OE$)

1. **Procesar** el conjunto de datos transaccionales brutos aplicando técnicas sistemáticas de codificación categórica, auditoría de duplicados y tratamiento de valores nulos mediante imputación robusta por mediana, enmarcando el diseño bajo las 5V del Big Data.
2. **Analizar** el comportamiento del cliente a través de estadística descriptiva, tablas de contingencia y matrices de correlación (evaluando variables de recencia, antigüedad, quejas y gasto) para aislar los factores determinantes en la deserción.
3. **Modelar** la probabilidad individual de deserción mediante el diseño e implementación comparativa de un clasificador lineal paramétrico (Regresión Logística con ponderación de clases) y un clasificador aditivo no paramétrico (XGBoost con optimización por gradiente).
4. **Evaluar** el rendimiento predictivo del sistema utilizando métricas de clasificación rigurosas (Sensibilidad/Recall, Precisión, F1-Score y área bajo la curva ROC-AUC), traduciendo los umbrales de decisión matemática en recomendaciones tácticas para la toma de decisiones empresariales.

---

## 4. Origen de los Datos

### 4.1 Estrategia de Selección y Datasets Candidatos
La propuesta metodológica está diseñada con una arquitectura flexible y agnóstica a la fuente de datos, contemplando dos alternativas para el desarrollo del proyecto:

* **Dataset Piloto / Base de Referencia Inicial (Prototipado Local):**
  * *Fuente:* Kaggle Repository (*E-Commerce Customer Churn Analysis and Prediction*).
  * *Volumen Muestral:* **5.630 perfiles de usuarios** únicos y 20 dimensiones analíticas.
  * *Rol en el Proyecto:* Se adopta como base de diseño inicial, exploración y validación matemática de laboratorio. Su valor reside en que ya cuenta con la variable objetivo `Churn` etiquetada y atributos muy específicos de fidelización (`Complain`, `Tenure`, `CashbackAmount`, `SatisfactionScore`), permitiendo validar la lógica algorítmica sin requerir infraestructura distribuida en la fase temprana.
  * *Condición de Desbalance:* Presenta una distribución asimétrica típica del comercio real (~83% retención vs ~17% abandono, relación ~5:1), lo cual justifica metodológicamente el uso de técnicas de compensación de clases en los algoritmos.

* **Dataset Candidato de Escala Masiva (Ampliación / Migración a Gran Escala):**
  * *Fuente:* Kaggle Repository (*Brazilian E-Commerce Public Dataset by Olist* - contemplado explícitamente en la guía del curso).
  * *Volumen Muestral:* **~100.000 órdenes de compra** y más de 112.000 registros transaccionales reales.
  * *Rol en el Proyecto:* Se establece como la alternativa directa de migración para la fase de implementación si se requiere someter el modelo a un orden de magnitud superior de datos. Permite cruzar tablas relacionales (clientes, geolocalización, reseñas de satisfacción, tiempos de entrega y métodos de pago) para construir un indicador de deserción basado en análisis RFM (Recencia, Frecuencia y Monetario) sobre transacciones auténticas de e-commerce.

### 4.2 Estructura y Diccionario de Variables de Referencia (Dataset Piloto)

| Variable | Tipo de Dato | Naturaleza | Rol en el Negocio |
| :--- | :--- | :--- | :--- |
| `CustomerID` | Numérico (Entero) | Identificador | Clave primaria de auditoría única por cliente (se excluirá del modelado predictivo). |
| `Churn` | Numérico (Binario) | Variable Objetivo | Estado del usuario: $1$ = Desertor (*Churn*), $0$ = Activo / Retenido. |
| `Tenure` | Numérico (Continuo) | Temporal | Meses acumulados de permanencia del cliente en la plataforma. |
| `PreferredLoginDevice` | Texto (Nominal) | Canal | Dispositivo principal de acceso (`Mobile Phone`, `Phone`, `Computer`). |
| `CityTier` | Numérico (Ordinal) | Demográfico | Nivel socioeconómico o de desarrollo urbano de la ciudad (1, 2, 3). |
| `WarehouseToHome` | Numérico (Continuo) | Logístico | Distancia en kilómetros entre el centro de distribución y el domicilio del comprador. |
| `PreferredPaymentMode` | Texto (Nominal) | Transaccional | Método de pago más utilizado (`Debit Card`, `Credit Card`, `UPI`, `Cash on Delivery`, `E-wallet`). |
| `Gender` | Texto (Binario) | Demográfico | Género declarado por el cliente (`Male`, `Female`). |
| `HourSpendOnApp` | Numérico (Continuo) | Interacción | Promedio de horas diarias de navegación dedicadas a la aplicación o sitio web. |
| `NumberOfDeviceRegistered`| Numérico (Discreto) | Comportamiento | Número de dispositivos autorizados sincronizados con la cuenta del usuario. |
| `PreferedOrderCat` | Texto (Nominal) | Comercial | Categoría preferente de compra (`Laptop & Accessory`, `Mobile Phone`, `Fashion`, `Grocery`, etc.). |
| `SatisfactionScore` | Numérico (Ordinal) | Percepción | Calificación del servicio asignada por el usuario en encuestas (escala 1 a 5). |
| `MaritalStatus` | Texto (Nominal) | Demográfico | Estado civil registrado (`Single`, `Married`, `Divorced`). |
| `NumberOfAddress` | Numérico (Discreto) | Logístico | Total de direcciones de envío configuradas en la cuenta. |
| `Complain` | Numérico (Binario) | Soporte | Registro formal de incidencias o reclamos en el último mes (1 = Sí, 0 = No). |
| `OrderAmountHikeFromlastYear`| Numérico (Continuo) | Financiero | Crecimiento porcentual del valor de compras respecto al año previo. |
| `CouponUsed` | Numérico (Discreto) | Comercial | Cantidad de cupones promocionales redimidos durante el periodo. |
| `OrderCount` | Numérico (Discreto) | Frecuencia | Número total de órdenes de compra confirmadas en la plataforma. |
| `DaySinceLastOrder` | Numérico (Continuo) | Recencia | Días transcurridos desde la última transacción confirmada. |
| `CashbackAmount` | Numérico (Continuo) | Fidelización | Promedio monetario de reembolsos o devoluciones recibidas en su billetera digital. |

---

## 5. Estrategia de Recolección y Pipeline de Procesamiento

### 5.1 Flujo de Ingesta y Desacoplamiento
Para garantizar la integridad y reproducibilidad del proyecto, se planifica una arquitectura por capas desacopladas:

```text
[Dataset Original Kaggle] 
       │
       ▼ (Extracción secundaria y desacoplamiento)
[data/raw/ecommerce_data.csv] ── (Lectura inmutable en Python / Pandas)
       │
       ▼ (Auditoría de duplicados + Imputación por mediana + One-Hot Encoding)
[data/processed/ecommerce_cleaned.csv] ── (Entrada para Modelado y Métricas)
```

1. **Capa Cruda (`data/raw/`):** Almacenará el archivo original en formato `.csv` delimitado, tratado como fuente de verdad inmutable (solo lectura).
2. **Capa Procesada (`data/processed/`):** Contendrá el dataset depurado tras ejecutar el protocolo de calidad: imputación de valores nulos mediante la **mediana**, corrección de inconsistencias léxicas (ej. homologar `Phone` con `Mobile Phone`) y tipado estricto.
3. **Control de Fuga de Información (*Data Leakage*):** Cualquier transformación de escala (`StandardScaler`) o cálculo de parámetros se ajustará exclusivamente sobre los datos de entrenamiento (*Train*), aplicándose posteriormente por transferencia sobre los datos de evaluación (*Test*).

### 5.2 Tecnologías y Entorno de Desarrollo
* **Lenguaje Principal:** Python.
* **Entorno de Ejecución:** Entorno virtual aislado (`.venv`) y cuadernos interactivos en **JupyterLab** (`notebooks/01_eda_limpieza.ipynb` y `notebooks/02_modelado_predictivo.ipynb`).
* **Librerías:**
  * *Manipulación y cálculo:* `pandas`, `numpy`, `openpyxl`.
  * *Exploración visual:* `matplotlib`, `seaborn`.
  * *Modelado estadístico y Machine Learning:* `scikit-learn`, `xgboost`.

---

## 6. Análisis de las 5V del Big Data

El proyecto se encuadra formalmente dentro de los principios rectores del Big Data:

* **Volumen (Perspectiva Crítica y Escalabilidad):**
  * *Dimensión del Prototipo Inicial:* El dataset piloto seleccionado comprende 5.630 registros y 20 variables (~112.600 celdas de datos). Con honestidad técnica y rigor de ingeniería, este tamaño representa una muestra analítica de laboratorio (*"Small Data"*), idónea para experimentación local rápida sin sobrecostos de infraestructura en la nube.
  * *Escalabilidad hacia Big Data Real:* En el comercio electrónico productivo, el volumen alcanza cientos de miles o millones de eventos diarios. Por ende, la propuesta contempla formalmente la migración hacia el dataset masivo de **Olist (~100.000 transacciones)** o repositorios corporativos. La formulación matemática vectorial y matricial de los algoritmos seleccionados (Regresión Logística y ensamble por gradiente en XGBoost) garantiza que el pipeline sea 100% extrapolable a motores de computación distribuida (como PySpark o Dask en clústeres) sin alterar la lógica del modelado.
* **Velocidad:** En el comercio electrónico, las interacciones suceden a ritmos vertiginosos: clics, carritos abandonados, transacciones y reclamos ocurren en tiempo real. La arquitectura del sistema separa el entrenamiento periódico por lotes (*batch*) del cálculo de inferencia probabilística en línea, permitiendo consultar el riesgo de abandono de un usuario en milisegundos mediante funciones vectorizadas.
* **Variedad:** El conjunto de datos presenta una notable diversidad tipológica: variables cuantitativas continuas (`DaySinceLastOrder`, `CashbackAmount`), discretas (`OrderCount`), variables cualitativas nominales (`PreferredPaymentMode`), binarias (`Complain`, `Gender`) y ordinales (`CityTier`, `SatisfactionScore`), exigiendo estrategias diferenciadas de codificación y preprocesamiento.
* **Veracidad:** La calidad de los datos es la piedra angular del modelado. El plan contempla una política estricta para resolver imperfecciones:
  * Eliminación de redundancias y errores tipográficos en variables de texto.
  * Imputación de datos faltantes mediante **mediana**, preservando la robustez estadística ante distribuciones asimétricas y valores extremos.
  * Auditoría de duplicados por identificador primario (`CustomerID`).
* **Valor:** Es la dimensión definitiva del proyecto. El valor se materializa en la reducción directa de la tasa de fuga de clientes y en la protección del flujo de ingresos futuros. Al anticipar con alta probabilidad qué usuarios están en riesgo de abandono, el departamento de fidelización puede accionar incentivos dirigidos (cupones, atención prioritaria), maximizando el Retorno de la Inversión (ROI) de las campañas de marketing.

---

## 7. Plan de Análisis Estadístico

El análisis estadístico se estructurará en dos etapas complementarias orientadas a responder preguntas concretas del negocio:

### 7.1 Análisis Univariado Proyectado
Se ejecutarán cálculos de estadística descriptiva para diagnosticar el comportamiento individual de cada variable:
* **Medidas de Tendencia Central y Posición:** Cálculo sistemático de media, mediana, cuartiles ($Q_1, Q_2, Q_3$) y percentiles extremos ($P_{95}, P_{99}$) para variables operativas (`Tenure`, `DaySinceLastOrder`, `CashbackAmount`).
* **Medidas de Dispersión y Forma:** Desviación estándar, Rango Intercuartílico ($IQR$) y coeficientes de asimetría (*skewness*) para detectar colas largas y sesgos distributivos en el comportamiento de compra.
* **Distribución de Frecuencias Categóricas:** Análisis de proporciones para modalidades de pago, categorías preferidas y dispositivos de acceso.

### 7.2 Análisis Bivariado y Contraste de Relaciones
Esta fase tiene como objetivo validar qué factores empujan al usuario a la fuga:
* **Análisis de Correlación Lineal y Monótona:** Cálculo de matrices de coeficientes de **Pearson** ($r$) y **Spearman** ($\rho$) entre todas las covariables numéricas y la variable objetivo `Churn`. Se espera corroborar una correlación inversa significativa entre la antigüedad (`Tenure`) y la deserción.
* **Tablas de Contingencia y Pruebas Chi-Cuadrado ($\chi^2$):** Evaluación de la independencia estadística entre variables categóricas críticas y el estado de fuga:
  * Contraste formal: ¿Existe dependencia estadística significativa entre el registro de una queja (`Complain = 1`) y la decisión de abandonar la plataforma (`Churn = 1`)?
* **Comparación de Grupos mediante Diagramas de Cajas (*Boxplots*):** Comparación de las distribuciones de recencia (`DaySinceLastOrder`) y gasto (`CashbackAmount`) segmentadas entre clientes retenidos y desertores, para cuantificar visual y numéricamente la brecha de inactividad previa al abandono.

---

## 8. Fundamentación y Modelado Matemático

Para resolver el problema de clasificación binaria supervisada, se formula un esquema de doble modelado: un modelo lineal probabilístico explicativo (*baseline*) y un modelo no lineal de ensamble por gradiente para capturar relaciones complejas.

### 8.1 Modelo Paramétrico Base: Regresión Logística Binaria
Dado un vector de características observadas $X = (x_1, x_2, \dots, x_p)^T \in \mathbb{R}^p$ correspondiente a un cliente, se modela la probabilidad a posteriori de abandono $P(Y = 1 \mid X)$ mediante la función logística estándar (sigmoide):

$$P(Y = 1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Donde el argumento logit $z$ corresponde a la combinación lineal de las covariables y sus respectivos coeficientes de ponderación:

$$z = \beta_0 + \sum_{j=1}^{p} \beta_j x_j = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$$

* **Propiedad de interpretabilidad:** El logit de las probabilidades (*log-odds*) se expresa de forma lineal:
  $$\ln\left(\frac{P(Y = 1 \mid X)}{1 - P(Y = 1 \mid X)}\right) = \beta_0 + \sum_{j=1}^{p} \beta_j x_j$$
  Esto permite estimar el factor de riesgo relativo (*Odds Ratio* = $e^{\beta_j}$), identificando exactamente cuánto aumenta o disminuye la propensión de abandono ante el incremento unitario de cualquier predictor.

### 8.2 Función de Pérdida y Tratamiento del Desbalance (Log-Loss Ponderada)
Dado el desbalance inherente de clases (~83% retención vs ~17% abandono), una función de costo simétrica convencional provocaría que el algoritmo prediga sistemáticamente a favor de la clase mayoritaria. Para neutralizar este sesgo, se implementará la **Entropía Cruzada Binaria Ponderada por Frecuencia Inversa**:

$$J(\beta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ w_1 \cdot y^{(i)} \ln(\hat{y}^{(i)}) + w_0 \cdot (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]$$

Donde $\hat{y}^{(i)} = \sigma(z^{(i)})$ y los pesos de penalización de cada clase se determinan en función de la muestra de entrenamiento:

$$w_1 = \frac{N}{2 \cdot N_{\text{churn}}}, \quad w_0 = \frac{N}{2 \cdot N_{\text{retenido}}}$$

Esta formulación penaliza severamente el error de clasificación cuando un cliente en fuga real ($y = 1$) no es detectado por el modelo.

### 8.3 Modelo Retador de Ensamble: XGBoost Classifier
Para modelar relaciones no lineales, efectos de umbral y patrones de interacción múltiple (ej. usuarios novatos con quejas y pedidos retrasados), se incorporará un ensamble aditivo de árboles de decisión mediante **eXtreme Gradient Boosting (XGBoost)**.

La predicción combinada tras $M$ etapas aditivas está dada por:

$$\hat{y}_i^{(M)} = \sum_{m=1}^{M} f_m(x_i), \quad f_m \in \mathcal{F}$$

Donde cada función $f_m$ es un árbol de decisión independiente. La función objetivo a minimizar en cada iteración $m$ incorpora un término de pérdida convexa más un término explícito de penalización de complejidad estructural:

$$\mathcal{L}^{(m)} \approx \sum_{i=1}^{N} \left[ g_i f_m(x_i) + \frac{1}{2} h_i f_m^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$

Donde:
* $g_i = \partial_{\hat{y}^{(m-1)}} l(y_i, \hat{y}^{(m-1)})$ es el gradiente de primer orden.
* $h_i = \partial^2_{\hat{y}^{(m-1)}} l(y_i, \hat{y}^{(m-1)})$ es el hessiano de segundo orden.
* $T$ representa el número de hojas terminales del árbol y $w_j$ los pesos asignados a cada hoja.
* $\gamma$ y $\lambda$ son parámetros de regularización que impiden el sobreajuste (*overfitting*).
* El desbalance se controlará directamente mediante el hiperparámetro de escala positiva:
  $$\text{scale\_pos\_weight} = \frac{N_{\text{retenido}}}{N_{\text{churn}}}$$

---

## 9. Propuesta de Visualización de Datos

Para que los resultados técnicos sean auditables y fácilmente interpretables por los líderes de negocio, se diseñará un panel gráfico compuesto por **cuatro visualizaciones clave**:

1. **Matriz de Confusión Comparada (Heatmaps interactivos):**
   * *Propósito:* Cuantificar de forma visual e inequívoca el número exacto de aciertos y errores en el conjunto de prueba (Verdaderos Positivos, Falsos Negativos, Verdaderos Negativos y Falsos Positivos).
   * *Aporte al negocio:* Permite a la gerencia auditar cuántos clientes desertores se lograrían salvar y cuál es la tasa de falsas alarmas que asumiría el equipo comercial.
2. **Curva ROC Comparativa (Receiver Operating Characteristic) y Área Bajo la Curva (AUC):**
   * *Propósito:* Graficar la Tasa de Verdaderos Positivos frente a la Tasa de Falsos Positivos a través de todos los umbrales de probabilidad posibles ($c \in [0, 1]$), contrastando la Regresión Logística frente a XGBoost y contra una referencia aleatoria ($AUC = 0.50$).
   * *Aporte al negocio:* Proporciona una medida independiente del umbral sobre la capacidad discriminativa intrínseca del sistema.
3. **Mapa de Calor de Correlaciones (Heatmap Multivariado):**
   * *Propósito:* Proyectar visualmente una cuadrícula con gradiente de color divergente que exhiba los coeficientes de correlación de todas las covariables numéricas contra `Churn`.
   * *Aporte al negocio:* Diagnosticar rápidamente multicolinealidad entre variables y validar visualmente los predictores con mayor fuerza de asociación.
4. **Gráfico de Importancia de Variables (*Feature Importance* y Coeficientes $|\beta|$):**
   * *Propósito:* Diagrama de barras horizontales ordenado jerárquicamente que destaque los factores de mayor ganancia de información (*information gain*) en XGBoost y la magnitud de impacto en la Regresión Logística.
   * *Aporte al negocio:* Explicar con transparencia **por qué** un cliente abandona la plataforma (identificando disparadores como quejas o inactividad prolongada).

---

## 10. Metodología de Evaluación y Resultados Esperados

### 10.1 Estrategia de Partición y Validación
* **División Estratificada (Train/Test Split):** Se plantea una partición estratificada del conjunto de datos en una proporción **80% entrenamiento** y **20% prueba independiente**.
  * *En el Dataset Piloto ($N \approx 5.630$):* Corresponde a $N_{\text{train}} \approx 4.504$ registros y $N_{\text{test}} \approx 1.126$ registros.
  * *En el Dataset Masivo Candidato ($N \approx 100.000$):* La misma partición representará $N_{\text{train}} \approx 80.000$ y $N_{\text{test}} \approx 20.000$ observaciones.
* **Garantía de Estratificación:** Se conservará rigurosamente la proporción de desertores en ambas particiones para evitar sesgos muestrales.
* **Aislamiento Técnico:** Ningún dato del conjunto de prueba intervendrá en la imputación de nulos, ni en la estandarización de escala (`StandardScaler`), garantizando una evaluación ciega y realista.

### 10.2 Métricas Clave y Criterios de Éxito del Negocio
A diferencia de los problemas de clasificación balanceados donde la Exactitud (*Accuracy*) es suficiente, en la predicción de churn esta métrica puede ser engañosa (un modelo trivial que prediga que "nadie se va" tendría un 83% de exactitud pero sería completamente inútil para el negocio).

Por ende, los criterios de evaluación y éxito metodológico se establecen en torno a:

| Métrica | Definición Matemática | Relevancia en el Negocio | Umbral Mínimo Esperado |
| :--- | :---: | :--- | :---: |
| **Sensibilidad (*Recall*)** | $\frac{TP}{TP + FN}$ | **Métrica Prioritaria.** Mide el porcentaje de clientes que realmente desertaron y que el modelo fue capaz de detectar a tiempo. Minimiza los Falsos Negativos (fugas no detectadas). | **$\ge 85.0\%$** |
| **Capacidad Discriminativa (*ROC-AUC*)** | $\int_0^1 \text{TPR}(FPR^{-1}(t)) \, dt$ | Capacidad global del clasificador para ordenar a los clientes asignando mayores probabilidades de riesgo a quienes verdaderamente desertarán. | **$\ge 88.0\%$** |
| **Puntaje F1 (*F1-Score*)** | $2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Media armónica entre la precisión y la exhaustividad, garantizando un balance operativo adecuado. | **$\ge 70.0\%$** |
| **Precisión (*Precision*)** | $\frac{TP}{TP + FP}$ | Porcentaje de acierto real cuando el modelo dispara una alerta de abandono, evitando el desgaste de recursos en falsas alarmas. | $\ge 55.0\%$ |

### 10.3 Protocolo de Matriz de Confusión y Análisis de Costos
La matriz de confusión se auditará bajo una matriz de costo asimétrico empresarial:
* **Costo del Falso Negativo ($FN$ - Cliente que se fuga sin ser detectado):** Máximo costo financiero. La empresa pierde el LTV del cliente, su margen transaccional recurrente y el costo histórico de haberlo adquirido.
* **Costo del Falso Positivo ($FP$ - Cliente fidelizado que recibe una alerta):** Costo financiero marginal. Se traduce únicamente en el costo menor de enviarle un cupón o correo de fidelización que de todas formas fortalece su relación con la marca.
* *Criterio de decisión:* Se priorizará deliberadamente la **Sensibilidad** sobre la Precisión pura, calibrando el umbral de decisión ($c$) para tolerar falsos positivos con tal de acorralar y minimizar los falsos negativos.

---

## 11. Conclusiones Esperadas e Impacto Operativo

### 11.1 Impacto Esperado en Marketing y Retención
La implementación planificada del modelo permitirá pasar de un esquema de retención reactivo (cuando el usuario ya cerró su cuenta) a una **estrategia de retención proactiva y automatizada**:

1. **Protocolo Inmediato ante Incidencias Técnicas:** Si el análisis confirma que las quejas (`Complain = 1`) y la baja antigüedad (`Tenure`) multiplican la probabilidad de fuga, se implementará una regla en el CRM: cualquier cliente con menos de 6 meses que radique una queja activará una alerta roja prioritaria (SLA de respuesta menor a 24 horas y oferta de compensación inmediata).
2. **Disparadores Automatizados por Recencia:** Programar campañas automáticas en la plataforma de correo/notificaciones cuando la variable `DaySinceLastOrder` se desvíe del patrón habitual del cliente, ofreciendo incentivos de envío gratis o puntos de fidelidad.
3. **Segmentación Estratégica por Niveles de Riesgo:**
   * **Riesgo Alto ($P(\text{churn}) \ge 0.70$):** Asignación de ejecutivos de cuenta, descuentos especiales agresivos y llamadas directas de soporte.
   * **Riesgo Moderado ($0.40 \le P(\text{churn}) < 0.70$):** Recomendaciones algorítmicas de productos basadas en su historial y beneficios de cashback.
   * **Riesgo Bajo ($P(\text{churn}) < 0.40$):** Flujo habitual de comunicaciones comerciales.

### 11.2 Limitaciones Identificadas y Hoja de Ruta de Escalamiento
* **Limitación Inicial del Muestreo:** El dataset piloto de 5.630 registros representa una muestra agregada de laboratorio con limitaciones para capturar micro-segmentos finos.
* **Hoja de Ruta hacia Big Data Masivo y Streaming:**
  1. *Fase de Escalamiento Muestral:* Migración del prototipo hacia el dataset de **Olist (~100.000 órdenes transaccionales)** o datasets tabulares a gran escala, enriqueciendo la ingeniería de características con variables de geolocalización, fletes logísticos y tiempos reales de entrega.
  2. *Fase de Streaming:* Despliegue en tiempo real utilizando arquitecturas distribuidas (Apache Kafka + Spark Streaming) para evaluar el riesgo de abandono durante la misma sesión de compra del usuario (por ejemplo, tras experimentar dos transacciones rechazadas consecutivas en la pasarela de pagos).

---

## 12. Referencias Bibliográficas

1. **Chen, T., & Guestrin, C.** (2016). *XGBoost: A Scalable Tree Boosting System*. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785
2. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer Series in Statistics. Springer New York. https://doi.org/10.1007/978-0-387-84858-7
3. **Provost, F., & Fawcett, T.** (2013). *Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking*. O'Reilly Media.
4. **Verbeke, W., Martens, D., Mues, C., & Baesens, B.** (2012). *Building comprehensible customer churn prediction models with advanced rule induction techniques*. IEEE Transactions on Knowledge and Data Engineering, 24(12), 2100–2113. https://doi.org/10.1109/TKDE.2011.177
5. **Zhu, B., Baesens, B., & Backiel, A.** (2018). *Benchmarking state-of-the-art techniques for customer churn prediction in telecom and retail*. Decision Support Systems, 107, 79–89. https://doi.org/10.1016/j.dss.2018.01.006
6. **Géron, A.** (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
7. **Verma, A.** (2021). *E-Commerce Customer Churn Analysis and Prediction Dataset*. Kaggle Repository. https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
8. **Olist & Kaggle.** (2018). *Brazilian E-Commerce Public Dataset by Olist*. Kaggle Repository. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
