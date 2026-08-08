"""
Charge le modèle entraîné et fait une prédiction sur un exemple.
Usage : python src/predict.py
"""

import joblib
from sklearn.datasets import load_iris


def main():
    model = joblib.load("models/model.pkl")
    data = load_iris()

    # Exemple : on prend la première fleur du dataset
    sample = data.data[0].reshape(1, -1)
    prediction = model.predict(sample)
    class_name = data.target_names[prediction[0]]

    print(f"Caractéristiques : {sample.tolist()}")
    print(f"Classe prédite : {class_name}")


if __name__ == "__main__":
    main()
