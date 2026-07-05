# Predicción del Éxito Académico - Caso de Estudio

Este repositorio contiene un pipeline de Machine Learning desarrollado para el Instituto Profesional Escuela de Comercio de Santiago. El objetivo del modelo es identificar tempranamente a los estudiantes en riesgo de bajo rendimiento académico.

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python 3.12+
* **Manipulación de Datos:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, GridSearchCV)
* **Visualización:** Matplotlib, Seaborn

## 📂 Estructura del Proyecto

* `data/`: Contiene los datos crudos y procesados (excluidos del control de versiones).
* `notebooks/`: Entornos interactivos de Jupyter para análisis exploratorio, Feature Engineering y evaluación del modelo.
* `src/`: Módulos de Python reutilizables (`data_cleaning.py`, `feature_eng.py`, `train_model.py`).
* `reports/figures/`: Gráficos generados por el modelo (matrices de confusión, curvas ROC, etc.).

## 🚀 Cómo ejecutar este proyecto

1. Clona el repositorio:
   ```bash
   git clone [https://github.com/tu-usuario/tu-repo.git](https://github.com/tu-usuario/tu-repo.git)
   ```

2. Instala las dependencias necesarias:
    ```bash
    pip install -r requirements.txt
    ```

3. Ejecuta los Jyper Notebooks ubicados en la carpeta `notebooks/` en orden secuencial (01, 02, 03)