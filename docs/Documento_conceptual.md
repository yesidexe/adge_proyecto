# Informe Metodológico y Técnico: Predicción de Abandono de Clientes (Customer Churn)

> **Documento de Sustentación Teórica y Metodológica**  
> **Área:** Analítica de Datos & Machine Learning Aplicado  
> **Sector:** Comercio Electrónico (*E-Commerce*)  
> **Estado:** Ejecutado y Validado con Datos Reales  

---

## 1. Contexto del Problema y Justificación Teórica

En los modelos de negocio basados en comercio electrónico y servicios digitales, la deserción de usuarios (**Churn**) es uno de los mayores destructores de rentabilidad. Adquirir un cliente nuevo puede costar entre **5 y 7 veces más** que retener a uno existente. 

### 1.1 El Costo Asimétrico del Error en la Detección de Churn
En problemas de clasificación tradicionales se suele priorizar la Exactitud (*Accuracy*). Sin embargo, en la predicción de fuga de clientes, los errores tienen costos financieros radicalmente distintos:
* **Falso Negativo (FN):** El modelo predice que el cliente es fiel, pero en realidad abandona la plataforma. El costo es muy alto: se pierde el valor de vida del cliente (*Lifetime Value* - LTV) y futuros ingresos por comisiones o recompras.
* **Falso Positivo (FP):** El modelo predice que un cliente va a desertar cuando en realidad pensaba quedarse. El costo es muy bajo: se le envía un cupón de descuento, cashback o un correo preventivo que refuerza su satisfacción.

Por esta razón, el diseño metodológico de este proyecto prioriza la **Sensibilidad (*Recall*)** y el **Área Bajo la Curva ROC (ROC-AUC)** sobre la simple exactitud global.

---

## 2. Fase 1: Adquisición y Comprensión del Dominio de Datos

El conjunto de datos corresponde a un registro transaccional estructurado de **5.630 perfiles de clientes** con **20 atributos** que cubren el ciclo de vida del comprador:

1. **Variables de Identificación:** `CustomerID` (código único de auditoría).
2. **Variable Objetivo:** `Churn` (variable binaria donde $1$ indica que el cliente abandonó la empresa en el último ciclo, y $0$ que permanece activo).
3. **Variables Temporales y de Antigüedad:**
   * `Tenure`: Meses de vinculación continua del usuario.
   * `DaySinceLastOrder`: Días transcurridos desde la última transacción realizada (Recencia).
4. **Variables de Soporte y Fricción:**
   * `Complain`: Registro de reclamos formales en el último mes (1 = Sí, 0 = No).
   * `SatisfactionScore`: Puntuación de percepción de servicio asignada por el cliente (escala 1 a 5).
5. **Variables de Comportamiento e Interacción Digital:**
   * `HourSpendOnApp`: Horas promedio diarias dedicadas a navegar en la app o sitio web.
   * `NumberOfDeviceRegistered`: Total de dispositivos vinculados a la cuenta.
   * `PreferredLoginDevice`: Dispositivo predominante de acceso (móvil o computador).
   * `WarehouseToHome`: Distancia logística en kilómetros desde el centro de distribución.
6. **Variables Transaccionales y de Fidelización:**
   * `OrderCount`: Total histórico de órdenes registradas.
   * `OrderAmountHikeFromlastYear`: Crecimiento porcentual del ticket de compra respecto al año anterior.
   * `CouponUsed`: Cupones de descuento redimidos.
   * `CashbackAmount`: Monto promedio reintegrado al cliente.
   * `PreferedOrderCat`: Categoría dominante de compras (tecnología, moda, alimentos).
   * `PreferredPaymentMode`: Método de pago habitual (tarjeta de débito, crédito, billetera digital, contraentrega).
7. **Variables Demográficas:** `Gender`, `MaritalStatus`, `CityTier`.

---

## 3. Fase 2: Calidad de Datos y Análisis Exploratorio (EDA)

Antes de cualquier modelado, se evaluó la integridad de la información mediante estadística descriptiva y pruebas de consistencia.

### 3.1 Tratamiento de Inconsistencias Categóricas
Se detectaron duplicidades semánticas generadas por variaciones en el registro de sistemas transaccionales:
* Dispositivos de acceso: `Phone` y `Mobile Phone` se unificaron bajo la categoría canónica `Mobile Phone`.
* Métodos de pago: Acrónimos como `CC` se integraron en `Credit Card`, y `COD` en `Cash on Delivery`.
* Categorías de compra: Se normalizó `Mobile` dentro de `Mobile Phone`.

### 3.2 Estrategia de Imputación de Valores Faltantes
Varias variables numéricas presentaban datos ausentes (`Tenure`, `DaySinceLastOrder`, `WarehouseToHome`, `OrderCount`, etc.).  
En lugar de eliminar filas (lo que reduciría la potencia estadística) o imputar con la media aritmética (la cual es sensible a valores extremos o distribuciones sesgadas), se aplicó **imputación por la mediana**:
$$\tilde{x} = \text{Mediana}(X_{\text{col}})$$
Esto preserva la representatividad central de los datos sin inflar la varianza ni crear valores artificialmente anómalos.

### 3.3 Hallazgos Principales del Comportamiento de Churn
Del análisis univariado y bivariado se extrajeron conclusiones cuantitativas directas:
* **Tasa Base de Deserción:** El **16.84%** de la base (948 clientes) desertó, frente al **83.16%** retenido (4.682 clientes). Existe un desbalance de clases moderado (~5:1).
* **El Efecto Multiplicador de las Quejas (`Complain`):**
  * Clientes sin reclamos recientes: Tasa de abandono del **10.8%**.
  * Clientes con reclamos recientes: Tasa de abandono del **31.7%**.  
  * *Conclusión:* Un reclamo insatisfecho casi **triplica** el riesgo de abandono inmediato.
* **Curva de Mortalidad Temprana (`Tenure`):** La gran mayoría de fugas se produce en los primeros 3 meses de vida del cliente. Los clientes que superan el primer año de permanencia muestran tasas de retención superiores al 90%.
* **Recencia (`DaySinceLastOrder`):** La inactividad prolongada muestra una relación monotónica creciente con la probabilidad de no retorno.

---

## 4. Fase 3: Ingeniería de Características (Feature Engineering)

Para alimentar a los modelos de machine learning se ejecutaron transformaciones formales:

### 4.1 Aislamiento de Identificadores
Se descartó la columna `CustomerID` de la matriz predictora $X$. Los identificadores artificiales no aportan información causal y provocan sobreajuste (*memorización de ruido*).

### 4.2 Codificación de Variables Categóricas (One-Hot Encoding)
Los algoritmos matemáticos requieren entradas numéricas continuas o booleanas. Se aplicó codificación en variables indicadoras binarias (*One-Hot Encoding*):
Para una variable con $k$ niveles, se generaron $k - 1$ columnas binarias (`drop_first=True`), evitando la colinealidad perfecta (trampa de la variable ficticia). La matriz final de características se expandió de 18 variables a **25 predictores numéricos**.

### 4.3 Partición Estratificada (Train / Test Split)
Se dividió el conjunto en:
* **80% Entrenamiento ($N = 4.504$ clientes):** Para el ajuste y aprendizaje de parámetros.
* **20% Prueba ($N = 1.126$ clientes):** Estrictamente reservado para evaluar la capacidad de generalización con datos nunca vistos.

Se aplicó **muestreo estratificado** con base en la variable objetivo (`stratify=y`), garantizando que tanto el set de entrenamiento como el de prueba preservaran idéntica proporción de desertores (16.8%).

### 4.4 Escalado y Prevención de Fuga de Información (*Data Leakage*)
Las variables como `CashbackAmount` (rango 0 a 300) y `Tenure` (rango 0 a 60) operan en órdenes de magnitud diferentes.
Se utilizó estandarización z-score:
$$z = \frac{x - \mu}{\sigma}$$
*Regla de rigor metodológico aplicada:* El escalador calculó la media $\mu$ y la desviación estándar $\sigma$ **únicamente sobre el conjunto de entrenamiento** (`fit_transform`) y aplicó dichos parámetros sobre el conjunto de prueba (`transform`), imposibilitando cualquier sesgo retrospectivo.

---

## 5. Fase 4: Fundamentación de los Modelos Predictivos

Se implementaron dos familias algorítmicas con propiedades complementarias:

```
                  Comparativa Estructural de Modelos
+------------------------------------+------------------------------------+
|        Regresión Logística         |              XGBoost               |
+------------------------------------+------------------------------------+
| - Modelo paramétrico lineal        | - Ensamble no paramétrico          |
| - Alta interpretabilidad (pesos β) | - Captura interacciones no lineales|
| - Rápido y sin sobreajuste severo  | - Estado del arte en datos tabulares|
+------------------------------------+------------------------------------+
```

### 5.1 Modelo Base: Regresión Logística Ponderada
La probabilidad posterior de que un cliente abandone ($Y=1$) dado su vector de atributos $X$ se expresa mediante la función sigmoide:
$$P(Y=1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-(\beta_0 + \sum_{j=1}^p \beta_j X_j)}}$$

#### Manejo de Desbalance: Ponderación de Clases
Dado que los clientes activos superan en proporción 5 a 1 a los desertores, una función de costo estándar ignoraría la clase minoritaria. Se incorporó una matriz de costo balanceada (`class_weight='balanced'`), la cual ajusta automáticamente los pesos de la función de pérdida inversamente proporcionales a las frecuencias de clase:
$$w_k = \frac{N}{2 \cdot N_k}$$
Esto penaliza con un factor ~5 veces mayor el error cuando el modelo se equivoca al predecir a un desertor real.

### 5.2 Modelo Avanzado: XGBoost Classifier (*Extreme Gradient Boosting*)
XGBoost construye de manera secuencial un ensamble de $M$ árboles de decisión donde cada nuevo árbol corrige los residuos de los anteriores:
$$\hat{y}_i^{(m)} = \hat{y}_i^{(m-1)} + f_m(x_i)$$

La optimización minimiza una función objetivo regularizada mediante expansiones de Taylor de segundo orden:
$$\mathcal{L}^{(m)} = \sum_{i=1}^N \left[ g_i f_m(x_i) + \frac{1}{2} h_i f_m^2(x_i) \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$
Donde $g_i$ y $h_i$ son el gradiente y el hessiano de la función de pérdida logarítmica, y $\gamma, \lambda$ penalizan la complejidad y profundidad de los árboles.

Para equilibrar el aprendizaje en el gradiente, se calibró el hiperparámetro de escala positiva:
$$\text{scale\_pos\_weight} = \frac{\sum (y_i = 0)}{\sum (y_i = 1)} = \frac{3.745}{759} \approx 4.94$$

---

## 6. Fase 5: Resultados Experimentales y Evaluación Comparativa

Ambos modelos fueron puestos a prueba sobre los **1.126 clientes del conjunto de prueba independiente**.

### 6.1 Tabla Comparativa de Rendimiento

| Modelo | Exactitud (*Accuracy*) | Sensibilidad (*Recall*) | Precisión (*Precision*) | F1-Score | Capacidad Discriminativa (*ROC-AUC*) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística** | 79.22% | **84.74%** | 43.99% | 57.91% | **88.52%** |
| **XGBoost Classifier** | **89.96%** | **90.00%** | **64.53%** | **75.16%** | **95.52%** |

### 6.2 Interpretación de las Métricas
1. **Sensibilidad (*Recall*):**
   * Regresión Logística detectó al **84.74%** de los clientes que se marcharon.
   * XGBoost detectó al **90.00%** de los clientes fugitivos (de 190 clientes reales que desertaron, XGBoost anticipó 171 a tiempo).
2. **Precisión:**
   * La Regresión Logística generó más falsas alarmas (precisión del 44%), mientras que XGBoost alcanzó un **64.53%**, lo que reduce el gasto innecesario en promociones.
3. **Área Bajo la Curva ROC (ROC-AUC):**
   * La Regresión Logística obtuvo un **0.885**, lo cual es catalogado como un clasificador excelente en la literatura.
   * XGBoost alcanzó un **0.955**, situándose en el rango sobresaliente cercano al óptimo para sistemas en producción.

### 6.3 Factores con Mayor Peso Predictivo (*Feature Importance*)
Al contrastar los coeficientes $\beta$ de la Regresión Logística y la ganancia de información en XGBoost, las variables dominantes fueron:
1. **Antigüedad (`Tenure`):** Coeficiente fuertemente negativo. Cada mes de permanencia reduce drásticamente la probabilidad de deserción.
2. **Registro de Quejas (`Complain`):** Coeficiente positivo de mayor magnitud. Es el principal catalizador de salida.
3. **Número de Direcciones Registradas (`NumberOfAddress`):** Cambios continuos de domicilio o múltiples registros incrementan la probabilidad de abandono.
4. **Recencia (`DaySinceLastOrder`):** Clientes con más de 15 a 20 días sin órdenes entran en la zona roja de riesgo.
5. **Categoría de Compra:** Compradores de tecnología y accesorios muestran patrones de mayor retención frente a categorías volátiles.

---

## 7. Fase 6: Estrategia de Retención Operativa y Recomendaciones

El valor del modelo reside en su capacidad para activar reglas de negocio preventivas:

1. **Protocolo de "Bandera Roja" ante Quejas:**
   * *Gatillo:* Cualquier usuario con `Tenure < 6 meses` que registre un reclamo (`Complain = 1`).
   * *Acción automática:* Asignar ticket a un agente sénior de retención con resolución garantizada en menos de 24 horas y emisión automática de un bono de compra.
2. **Activación de Campañas de Reactivación por Recencia:**
   * *Gatillo:* Clientes cuyo `DaySinceLastOrder` supere el umbral de 10 días de inactividad respecto a su frecuencia habitual.
   * *Acción automática:* Notificación push personalizada recomendando productos con descuento en su categoría favorita (`PreferedOrderCat`).
3. **Intervención Basada en Score de Probabilidad:**
   * **Riesgo Alto ($P > 0.70$):** Ofrecer incentivos monetarios directos (cashback inmediato o envío gratuito en la siguiente compra).
   * **Riesgo Medio ($0.40 \le P \le 0.70$):** Comunicaciones de valor de marca, encuestas de satisfacción o recomendaciones basadas en historial.
   * **Riesgo Bajo ($P < 0.40$):** Flujo de compras estándar sin inversión en descuentos preventivos.

---

## 8. Trazabilidad de Artefactos del Proyecto

* **Datos fuente y procesados:**
  * Datos brutos: [`data/raw/ecommerce_data.csv`](file:///workspaces/adge_proyecto/data/raw/ecommerce_data.csv)
  * Datos limpios e imputados: [`data/processed/ecommerce_cleaned.csv`](file:///workspaces/adge_proyecto/data/processed/ecommerce_cleaned.csv)
* **Notebooks de experimentación:**
  * Notebook 01: [`notebooks/01_eda_limpieza.ipynb`](file:///workspaces/adge_proyecto/notebooks/01_eda_limpieza.ipynb) (Calidad de datos y EDA).
  * Notebook 02: [`notebooks/02_modelado_predictivo.ipynb`](file:///workspaces/adge_proyecto/notebooks/02_modelado_predictivo.ipynb) (Modelos, ROC, matrices y comparativa).
