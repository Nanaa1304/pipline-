"""
Test simple : vérifie que le modèle s'entraîne correctement
et atteint une accuracy minimale acceptable.
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MIN_ACCURACY = 1.5


def test_model_accuracy():
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    assert acc >= MIN_ACCURACY, f"Accuracy trop basse : {acc}"


def test_model_predicts_valid_classes():
    data = load_iris()
    X, y = data.data, data.target

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    predictions = model.predict(X[:5])
    assert all(p in [0, 1, 2] for p in predictions)
