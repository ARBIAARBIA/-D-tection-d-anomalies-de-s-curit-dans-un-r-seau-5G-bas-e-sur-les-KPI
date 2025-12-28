# model.py

from sklearn.ensemble import IsolationForest
import numpy as np

def train_isolation_forest(X):
    """
    Entraînement robuste du modèle Isolation Forest
    """

    model = IsolationForest(
        n_estimators=200,          # Plus d'arbres = plus stable
        max_samples="auto",
        contamination=0.05,        # 5% d'anomalies supposées
        random_state=42,
        n_jobs=-1                  # Utilise tous les cœurs CPU
    )

    model.fit(X)
    return model


def predict_anomalies(model, X):
    """
    Prédiction des anomalies + score d'anomalie
    """
    predictions = model.predict(X)      # -1 anomalie | 1 normal
    scores = model.decision_function(X) # Score de normalité

    return predictions, scores
