import pandas as pd
import numpy as np

engineer_features = lambda df: df.assign(
    indice_eficiencia_academica = df['num__promedio_notas'] / (np.abs(df['skewed__minutos_totales_lms']) + 0.1),
    
    riesgo_sobrecarga = (3.0 - df['num__tasa_asistencia']) * (
        (df.get('nominal__estado_laboral_Full-time', 0.0) * 1.5) +
        (df.get('nominal__estado_laboral_Part-time', 0.0) * 1.2) +
        (df.get('nominal__estado_laboral_Inactivo', 0.0) * 1.0)
    ),
    
    ratio_compromiso_asincrono = df['skewed__minutos_totales_lms'] / (df['num__tasa_asistencia'] + 4.0)
)