# preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path):
    """
    Prétraitement robuste des données KPI 5G :
    - Chargement
    - Nettoyage
    - Gestion des valeurs manquantes
    - Normalisation
    """

    # 1. Chargement
    df = pd.read_csv(csv_path)

    # 2. Sélection des colonnes numériques uniquement
    df_numeric = df.select_dtypes(include=[np.number])

    # 3. Gestion des valeurs manquantes (remplacement par la médiane)
    df_numeric = df_numeric.fillna(df_numeric.median())

    # 4. Suppression des colonnes constantes (variance nulle)
    df_numeric = df_numeric.loc[:, df_numeric.var() > 0]

    # 5. Normalisation (StandardScaler)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_numeric)

    return df, df_numeric, data_scaled
