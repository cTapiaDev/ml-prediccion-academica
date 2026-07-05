import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier

prepare_training_data = lambda df: df.assign(
    exito_academico=np.array([1, 0, 1, 0, 1, 1][:len(df)])
)

def plot_data_split(y_orig, y_train, y_test):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    sns.countplot(x=y_orig, hue=y_orig, ax=ax[0], palette='pastel', legend=False).set_title('Dataset Original')
    sns.countplot(x=y_train, hue=y_train, ax=ax[1], palette='Set2', legend=False).set_title('Entrenamiento')
    sns.countplot(x=y_test, hue=y_test, ax=ax[2], palette='Set1', legend=False).set_title('Prueba')
    
    fig.suptitle('Distribución Estratificada de Clases (0: Riesgo, 1: Éxito)', fontsize=14)
    for axes in ax:
        axes.set_xlabel('Clase')
        axes.set_ylabel('Cantidad')
    
    os.makedirs('../reports/figures', exist_ok=True)
    plt.savefig('../reports/figures/train_test_split.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_random_forest(X_train, y_train):
    rf_classifier = RandomForestClassifier(random_state=42)
    param_grid = {
        'max_depth': [3, 5, 10],          
        'n_estimators': [50, 100],    
        'min_samples_split': [2, 5]    
    }
    
    cv_strategy = KFold(n_splits=2, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf_classifier, 
        param_grid=param_grid, 
        cv=cv_strategy, 
        scoring='accuracy',
        n_jobs=-1 
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search