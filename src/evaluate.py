from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import joblib

iris = load_iris()
X, y = iris.data, iris.target

model = joblib.load("iris_model.joblib")
y_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
