import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer

def build_preprocessor():
    log_transformer = FunctionTransformer(lambda x: np.log1p(x), validate=True, feature_names_out='one-to-one')

    num_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    skewed_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log', log_transformer),
        ('scaler', StandardScaler())
    ])

    nom_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    ord_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[['Basica', 'Media', 'Superior']]))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, ['promedio_notas', 'tasa_asistencia']),
        ('skewed', skewed_pipeline, ['minutos_totales_lms']),
        ('nominal', nom_pipeline, ['carrera', 'jornada', 'estado_laboral']),
        ('ordinal', ord_pipeline, ['educacion_padres'])
    ])

    return preprocessor