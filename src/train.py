"""
Entraîne un modèle simple de classification sur le dataset Iris
et sauvegarde le modèle entraîné dans models/model.pkl
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    # 1. Charger les données
    data = load_iris()
    X, y = data.data, data.target

    # 2. Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Entraîner le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Évaluer
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy sur le jeu de test : {acc:.4f}")

    # 5. Sauvegarder le modèle
    joblib.dump(model, "models/model.pkl")
    print("Modèle sauvegardé dans models/model.pkl")


if __name__ == "__main__":
    main()
