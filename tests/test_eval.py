import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = joblib.load("iris_model.joblib")
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc >= 0.9, f"Model accuracy too low: {acc}"
