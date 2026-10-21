"""Gráficos y reporte en Markdown que documentan cómo se entrenó el modelo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: funciona en servidores y en tests
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from churn.modeling import LOGISTIC, XGBOOST  # noqa: E402

COLORS = {LOGISTIC: "#2b5c8f", XGBOOST: "#2ca02c"}
TOP_N_FEATURES = 10


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_roc_curves(y_true: Sequence[int], probs: Mapping[str, np.ndarray], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        ax.plot(fpr, tpr, lw=2, color=COLORS.get(name), label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Azar (AUC = 0,50)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos (Recall)")
    ax.set_title("Curvas ROC en el conjunto de prueba", weight="bold")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_confusion_matrices(test_metrics: Mapping[str, Mapping[str, float]], path: Path) -> Path:
    fig, axes = plt.subplots(1, len(test_metrics), figsize=(6 * len(test_metrics), 5), squeeze=False)
    for ax, (name, m) in zip(axes[0], test_metrics.items()):
        matrix = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:,}".replace(",", "."), ha="center", va="center", weight="bold",
                    color="white" if value > matrix.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred: Retenido", "Pred: Churn"])
        ax.set_yticks([0, 1], ["Real: Retenido", "Real: Churn"])
        ax.set_title(f"{name}\nRecall {m['recall']:.1%} | Precisión {m['precision']:.1%}", weight="bold")
    return _save(fig, path)


def plot_feature_importance(importances: Mapping[str, pd.Series], path: Path) -> Path:
    fig, axes = plt.subplots(1, len(importances), figsize=(7 * len(importances), 5), squeeze=False)
    for ax, (name, series) in zip(axes[0], importances.items()):
        top = series.sort_values(ascending=False).head(TOP_N_FEATURES)[::-1]
        ax.barh(top.index, top.values, color=COLORS.get(name))
        ax.set_title(f"Top {TOP_N_FEATURES} variables — {name}", weight="bold")
        ax.set_xlabel("Importancia" if name == XGBOOST else "|Coeficiente|")
    return _save(fig, path)


def plot_segments(profile: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [f"Seg. {i}" for i in profile.index]
    axes[0].bar(labels, profile["clientes"], color="#2b5c8f")
    axes[0].set_title("Clientes por segmento", weight="bold")
    if "tasa_churn" in profile:
        axes[1].bar(labels, profile["tasa_churn"] * 100, color="#d9534f")
        axes[1].set_title("Tasa de churn real por segmento (%)", weight="bold")
    return _save(fig, path)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{value * 100:.2f} %"


def _n(value: int) -> str:
    """Entero con separador de miles estilo español (5.630)."""
    return f"{value:,}".replace(",", ".")


def write_training_report(path: Path, ctx: Mapping[str, Any]) -> Path:
    """Escribe `training_report.md` explicando datos, preprocesamiento, modelos y resultados."""
    data, split = ctx["data"], ctx["split"]
    cv, test = ctx["cv"], ctx["test"]
    segmentation = ctx["segmentation"]
    profile: pd.DataFrame = segmentation["profile"]

    nulls = ", ".join(f"`{c}` ({n})" for c, n in data["nulls"].items()) or "ninguno"
    fixes = "; ".join(
        f"`{col}`: " + ", ".join(f"`{a}` → `{b}`" for a, b in mapping.items())
        for col, mapping in data["category_fixes"].items()
    )
    hyper_rows = [(name, f"`{params}`") for name, params in ctx["hyperparameters"].items()]
    cv_rows = [
        (name, *(f"{m[f'{k}_mean']:.3f} ± {m[f'{k}_std']:.3f}" for k in ("recall", "precision", "f1", "roc_auc")))
        for name, m in cv.items()
    ]
    test_rows = [
        (name + (" ⭐" if name == ctx["selected"] else ""), _pct(m["recall"]), _pct(m["precision"]),
         _pct(m["f1"]), _pct(m["roc_auc"]), _pct(m["accuracy"]), m["tp"], m["fn"], m["fp"], m["tn"])
        for name, m in test.items()
    ]
    feature_blocks = []
    for name, items in ctx["top_features"].items():
        rows = [(i + 1, f"`{feat}`", f"{val:.4f}") for i, (feat, val) in enumerate(items)]
        feature_blocks.append(f"**{name}**\n\n" + _md_table(["#", "Variable", "Peso"], rows))
    seg_rows = []
    for segment, row in profile.iterrows():
        seg_rows.append(
            (segment, _n(int(row["clientes"])), _pct(row["tasa_churn"]) if "tasa_churn" in row else "-",
             segmentation["descriptions"][int(segment)])
        )
    silhouettes = ", ".join(f"k={k}: {v:.3f}" for k, v in segmentation["silhouette_by_k"].items())

    text = f"""# Reporte de entrenamiento — Predicción de churn

Generado automáticamente por `python -m churn.train` el {ctx['generated_at']}.
Librerías: {ctx['versions']}.

## 1. Datos

- Origen: `{data['source']}` (archivo original, no se modifica), **{_n(data['n_rows'])} clientes × {data['n_cols']} columnas**.
- Variable objetivo `Churn`: {_n(data['n_churn'])} clientes desertores ({_pct(data['churn_rate'])}) → clases **desbalanceadas**.
- IDs duplicados: {data['duplicated_ids']}.
- Valores nulos por columna: {nulls}.
- Limpieza de etiquetas duplicadas: {fixes}.

## 2. Preprocesamiento

1. Partición **{split['train_pct']:.0f} % / {split['test_pct']:.0f} %** estratificada (`random_state={ctx['random_state']}`): {_n(split['n_train'])} clientes de entrenamiento ({_pct(split['churn_train'])} churn) y {_n(split['n_test'])} de prueba ({_pct(split['churn_test'])} churn).
2. Dentro de un `Pipeline` (ajustado **solo con entrenamiento**, sin *data leakage*):
   - Numéricas: imputación por **mediana** y, para la regresión logística, estandarización (`StandardScaler`).
   - Categóricas: **One-Hot Encoding** (categorías nuevas se ignoran al predecir).
3. `CustomerID` y `Churn` no entran como variables predictoras.

> A diferencia del notebook 01, la mediana se calcula **después** de separar entrenamiento y prueba, por lo que
> las métricas pueden diferir levemente de las del `README.md` (ver `notebooks/03_pipeline_reproducible.ipynb`).

## 3. Modelos

{_md_table(['Modelo', 'Hiperparámetros'], hyper_rows)}

- **Regresión Logística** (base): interpretable; `class_weight='balanced'` penaliza más no detectar a un desertor.
- **XGBoost** (avanzado): ensamble de árboles por gradiente que captura relaciones no lineales; `scale_pos_weight` = retenidos / desertores del entrenamiento.

## 4. Validación cruzada (solo sobre entrenamiento, {ctx['cv_folds']} particiones estratificadas)

{_md_table(['Modelo', 'Recall', 'Precisión', 'F1', 'ROC-AUC'], cv_rows)}

## 5. Resultados en prueba (N = {_n(split['n_test'])})

{_md_table(['Modelo', 'Recall', 'Precisión', 'F1', 'ROC-AUC', 'Accuracy', 'TP', 'FN', 'FP', 'TN'], test_rows)}

**Modelo seleccionado: {ctx['selected']}** — criterio: {ctx['selection_rule']}. Se prioriza el Recall porque no detectar a un cliente que se va cuesta más que enviar una promoción de más. Umbral de decisión: {ctx['decision_threshold']}.

![Curvas ROC](figures/roc_curves.png)
![Matrices de confusión](figures/confusion_matrices.png)

### Variables más influyentes

{(chr(10) * 2).join(feature_blocks)}

![Importancia de variables](figures/feature_importance.png)

## 6. Segmentación de clientes (K-Means, sin usar la etiqueta `Churn`)

Variables: {', '.join(f'`{f}`' for f in segmentation['features'])} (imputación + estandarización). Se probó cada `k` de 3 a 6 (rango acotado a propósito: k = 2 solo separaría dos grandes grupos) y se eligió el de mayor coeficiente de silueta ({silhouettes}) → **k = {segmentation['k']}**.

{_md_table(['Segmento', 'Clientes', 'Churn real', 'Perfil (vs promedio)'], seg_rows)}

![Segmentos](figures/segments.png)

## 7. Cómo reproducirlo

```bash
pip install -r requirements.txt && pip install -e .
python -m churn.train                      # entrena, evalúa y regenera modelos, reportes y datasets
python -m churn.predict --input data/raw/ecommerce_data.csv --output predicciones.csv
pytest --cov=src --cov-report=term-missing
```

## 8. Archivos generados

| Archivo | Contenido |
| --- | --- |
| `models/churn_model.joblib` | Pipeline completo (preprocesamiento + modelo seleccionado) |
| `models/segmenter.joblib` | Pipeline de segmentación K-Means |
| `models/model_card.json` | Ficha del modelo: hiperparámetros, versiones, métricas |
| `reports/metrics.json` | Todas las métricas (prueba y validación cruzada) |
| `data/processed/test_predictions.csv` | Probabilidades de cada modelo sobre el conjunto de prueba |
| `data/processed/customer_segments.csv` | Segmento asignado a cada cliente |

## 9. Limitaciones

- Una sola partición de prueba; la validación cruzada solo se calcula sobre entrenamiento.
- Umbral fijo de {ctx['decision_threshold']}; las bandas de riesgo (Bajo/Medio/Alto) son heurísticas, no probabilidades calibradas.
- El modelo guardado se entrena con el {split['train_pct']:.0f} % de los clientes y se evalúa con el {split['test_pct']:.0f} % restante; el segmentador usa el 100 % porque no depende de la etiqueta.
- Dataset piloto de {_n(data['n_rows'])} clientes: validar con datos reales antes de usarlo en producción.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
