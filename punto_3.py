import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Carga el archivo CSV (asegúrate de que el archivo "EmployeesData.csv" esté en el mismo directorio)
df = pd.read_csv("EmployeesData.csv")

# Paso 1: Identificar la columna objetivo y separarla
y = df["LeaveOrNot"]
X = df.drop(columns=["LeaveOrNot"])

# Paso 2: Codificación de variables categóricas
X_encoded = pd.get_dummies(X)

# Paso 3: Partición estratificada del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)

# Paso 4: Entrenamiento de los modelos Random Forest
model_normal = RandomForestClassifier(random_state=42)
model_balanced = RandomForestClassifier(class_weight="balanced", random_state=42)

model_normal.fit(X_train, y_train)
model_balanced.fit(X_train, y_train)

# Paso 5: Métricas de desempeño
y_pred_normal = model_normal.predict(X_test)
y_pred_balanced = model_balanced.predict(X_test)

accuracy_normal = accuracy_score(y_test, y_pred_normal)
accuracy_balanced = accuracy_score(y_test, y_pred_balanced)

f1_normal = f1_score(y_test, y_pred_normal)
f1_balanced = f1_score(y_test, y_pred_balanced)

conf_matrix_normal = confusion_matrix(y_test, y_pred_normal)
conf_matrix_balanced = confusion_matrix(y_test, y_pred_balanced)

# Paso 6: Comparación de rendimiento
print(f"Accuracy (Normal): {accuracy_normal:.4f}")
print(f"Accuracy (Balanced): {accuracy_balanced:.4f}")

print(f"F1 Score (Normal): {f1_normal:.4f}")
print(f"F1 Score (Balanced): {f1_balanced:.4f}")

print("Matriz de confusión (Normal):")
ConfusionMatrixDisplay(conf_matrix_normal).plot()

print("Matriz de confusión (Balanced):")
ConfusionMatrixDisplay(conf_matrix_balanced).plot()
