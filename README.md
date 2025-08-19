# Desafío 3 — Predicción de Cancelación (Churn) · Telecom X

**Descripción breve**
Este repositorio contiene el flujo completo para predecir la probabilidad de cancelación (churn) de clientes de Telecom X. Incluye desde la carga y preparación de los datos hasta el entrenamiento, optimización y evaluación de modelos de Machine Learning, junto con herramientas para interpretar y poner en producción el modelo más efectivo.

---

## 1. Objetivo del proyecto

Construir un pipeline reproducible que permita:

* Identificar clientes con alta probabilidad de cancelar en el corto plazo.
* Entregar métricas y explicaciones que ayuden a priorizar acciones de retención.
* Proveer artefactos (modelos, reportes y datos procesados) listos para integrar en procesos operativos.

---

## 2. Estructura del repositorio

```
challenge3-telecomx/
├─ data/
│  ├─ raw/                      # Datos originales (no tocar)
│  └─ processed/                # Dataset limpio listo para modelado (telecom_churn_clean.csv)
├─ notebooks/                   # Notebooks exploratorios y de validación
├─ src/
│  ├─ data_pipeline.py          # Carga, limpieza y transformaciones reproducibles
│  ├─ features.py               # Ingeniería de variables
│  ├─ train.py                  # Entrena y guarda modelos (con CV y tuning)
│  ├─ evaluate.py               # Evaluación y generación de reportes/plots
│  └─ explainability.py         # SHAP / importancias y reportes interpretables
├─ models/                      # Modelos entrenados (.pkl / .joblib)
├─ reports/                     # Imágenes y reportes (PNG, HTML)
├─ requirements.txt
├─ README.md                    # Este documento
└─ LICENSE
```

---

## 3. Datos

**Archivo principal:** `data/processed/telecom_churn_clean.csv` (resultado de la fase de limpieza).

**Descripción:** contiene variables demográficas, de servicio (contrato, internet, teléfono), cargos (`account_Charges_Monthly`, `account_Charges_Total`), tenure, y la variable objetivo `Churn` (0/1).

**Nota:** si no cuentas con el CSV, ejecuta `src/data_pipeline.py` para regenerarlo desde `data/raw/` (o desde la URL original si procede).

---

## 4. Requisitos e instalación

Recomendado: crear un entorno virtual antes de instalar dependencias.

```bash
python -m venv venv
# Windows:
# .\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt (ejemplo)**

```
pandas>=1.5
numpy>=1.24
scikit-learn>=1.2
matplotlib>=3.5
seaborn>=0.12
lightgbm>=4.0   # opcional pero recomendado
shap>=0.42      # para interpretabilidad
joblib
jupyterlab      # si quieres ejecutar notebooks
```

---

## 5. Flujo de ejecución (rápido)

1. **Preprocesar y limpiar**

```bash
python src/data_pipeline.py --input data/raw/telecom_raw.json --output data/processed/telecom_churn_clean.csv
```

2. **Generar features (opcional)**

```bash
python src/features.py --input data/processed/telecom_churn_clean.csv --output data/processed/telecom_features.csv
```

3. **Entrenar modelos y optimizar hiperparámetros**

```bash
python src/train.py --data data/processed/telecom_features.csv --models_out models/
```

4. **Evaluar y generar reportes**

```bash
python src/evaluate.py --models models/ --data data/processed/telecom_features.csv --out reports/
```

Cada script imprime un resumen y genera artefactos (figuras PNG, métricas en JSON y modelos serializados).

---

## 6. Modelos incluidos y breve explicación

Se recomiendan probar al menos los siguientes modelos:

* **Regresión Logística** — Base interpretable; útil como baseline.
* **Random Forest** — Buen rendimiento con poca ingeniería, robusto a outliers.
* **LightGBM** — Boosting eficiente; suele ofrecer mejor rendimiento en problemas tabulares.

**Estrategia de validación:** usar `StratifiedKFold` con búsqueda de hiperparámetros (GridSearchCV o RandomizedSearchCV) optimizando `roc_auc` o `average_precision` según el objetivo.

---

## 7. Métricas y criterios de aceptación

* **ROC AUC** (discriminación completa).
* **Precision / Recall / F1** (especial atención a recall/precision en la clase positiva si se prioriza identificar churners).
* **PR-AUC** si la clase positiva está desbalanceada.
* **Métricas operativas:** % reducción esperada de churn tras intervención (estimada por uplift o pruebas A/B).

---

## 8. Interpretabilidad y explicación de predicciones

Se sugiere generar:

* **Importancias globales** (feature\_importances\_ / coeficientes normalizados).
* **Explicaciones locales** con SHAP para entender por cliente por qué el modelo predice alto riesgo.

Estos reportes permiten diseñar acciones dirigidas y justificar campañas frente a stakeholders.

---

## 9. Resultados esperados y artefactos

Al ejecutar el pipeline completo se deben generar:

* `models/best_model.pkl` — modelo serializado listo para scoring.
* `reports/metrics.json` — métricas agregadas de validación y test.
* `reports/feature_importance.png` — ranking de variables.
* `reports/roc_curve.png`, `reports/confusion_matrix.png` — diagnósticos de rendimiento.

---

## 10. Buenas prácticas y reproducibilidad

* Fijar `random_state` en todas las etapas (split, modelos, CV).
* Versionar datos en `data/raw/` y preservar `data/processed/` para trazabilidad.
* Guardar el `requirements.txt` con versiones exactas si necesitas recrear el entorno.
* Registrar experimentos con herramientas como MLflow o Weights & Biases para comparaciones reproducibles.

---

## 11. Despliegue (sugerencia rápida)

* **Scoring batch:** programar scoring diario/semanal que cargue nuevos clientes y genere una lista con `probabilidad_churn` > umbral para acciones de retención.
* **API de scoring:** empaquetar el modelo en un microservicio (FastAPI / Flask) que devuelva probabilidades para integrarlo con CRM.
* **Observabilidad:** monitorizar distribución de predicciones, drift y metricas operativas (TPR, FPR over time).

---

## 12. Limitaciones y aspectos a mejorar

* Correlación ≠ causalidad: las acciones deben validarse con A/B tests.
* Variables faltantes como historial de tickets, tiempos de reparación o NPS pueden mejorar mucho la predicción.
* LightGBM/GBDT pueden sobreajustar si no se regulan apropiadamente; usar early\_stopping y validación.

---

## 13. Próximos pasos recomendados

* Implementar un experimento piloto de retención (scoring → intervención → evaluación).
* Añadir fuentes operacionales (tickets, logs de red).
* Automatizar CI/CD para retrain mensual y monitorización del desempeño del modelo.

---

Autor: Gerardo Gonzalez Benitez — Proyecto desarrollado como desafío de Telecom X / Alura.

---

¿Quieres que convierta este README en `README.md` en el repositorio y añada un badge con los resultados del mejor modelo (ROC AUC)?
