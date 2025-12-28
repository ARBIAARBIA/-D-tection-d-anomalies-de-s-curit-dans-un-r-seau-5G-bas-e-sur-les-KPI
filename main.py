# main.py
# - Chargement des KPI 5G
# - Prétraitement des données
# - Entraînement du modèle IA (Isolation Forest)
# - Détection et affichage des anomalies

# Module de prétraitement des données
from preprocess import load_and_preprocess_data

# Module IA : entraînement et prédiction des anomalies
from model import train_isolation_forest, predict_anomalies

def main():
    """
    Fonction principale du pipeline de détection d'anomalies.
    Elle orchestre toutes les étapes du projet :
    - Prétraitement
    - Entraînement du modèle
    - Détection des anomalies
    """
    print("🔄 Chargement et prétraitement des données...")
    df, df_numeric, X_scaled = load_and_preprocess_data("kpi_5g.csv")

    print("🤖 Entraînement du modèle Isolation Forest...")
    model = train_isolation_forest(X_scaled)

    print("🚨 Détection des anomalies...")
    predictions, scores = predict_anomalies(model, X_scaled)

    df["anomaly"] = predictions
    df["anomaly_score"] = scores #anomaly_score : score de normalité (plus bas = plus anormal)

    anomalies = df[df["anomaly"] == -1] # Extraction des anomalies détectées

    print(f"Nombre total d'échantillons : {len(df)}")
    print(f"Nombre d'anomalies détectées : {len(anomalies)}")

    print("\nExemples d'anomalies :")
    print(anomalies.head())

if __name__ == "__main__":
    main()
