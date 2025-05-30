from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Paso 1: Carga del conjunto de datos
models_dir = "../models/"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Cargar los datos
url = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
df = pd.read_csv(url)

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 2: Construye un boosting
learning_rates = [0.01, 0.1, 0.5, 1.0]
n_estimators_range = [50, 100, 150]
results = {}

for lr in learning_rates:
    for n in n_estimators_range:
        model = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[(lr, n)] = accuracy_score(y_test, y_pred)




best_lr = max(results, key=results.get)[0]
best_n = max(results, key=results.get)[1]
best_model = GradientBoostingClassifier(n_estimators=best_n, learning_rate=best_lr, random_state=42)
best_model.fit(X_train, y_train)

y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Exactitud del modelo Boosting: {accuracy_best:.2f}")

# Paso 3: Guarda el modelo
model_path = os.path.join(models_dir, "boosting_classifier.pkl")

with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

print(f"Modelo guardado correctamente en {model_path}")

# Paso 4: Analiza y compara los resultados de los modelos
print("\nReporte de clasificación para Boosting:")
print(classification_report(y_test, y_pred_best))

tree_acc = 0.76  # Accuracy reportado del modelo de árbol de decisión
rf_acc = 0.71  # Accuracy reportado del modelo Random Forest
boosting_acc = accuracy_best

models_comparison = {"Decision Tree": tree_acc, "Random Forest": rf_acc, "Boosting": boosting_acc}



print(f"Accuracy - Árbol de Decisión: {tree_acc:.2f}")
print(f"Accuracy - Random Forest: {rf_acc:.2f}")
print(f"Accuracy - Boosting: {boosting_acc:.2f}")