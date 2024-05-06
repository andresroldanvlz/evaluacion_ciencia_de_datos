import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
# Cargar el archivo CSV
seguros = pd.read_csv("EmployeesData.csv")
df = pd.read_csv("EmployeesData.csv")

# Crear subplots con histograma y gráfico de torta para la distribución de niveles de estudio
fig, axs = plt.subplots(1, 2, figsize=(12, 6))


# Gráfico de torta
nivel_counts = df["Education"].value_counts()
axs[1].pie(nivel_counts, labels=nivel_counts.index, autopct='%1.1f%%', startangle=90)
axs[1].set_title('Distribución de Niveles de Estudio')


# Cargar el archivo CSV
seguros = pd.read_csv("EmployeesData.csv")
df = pd.read_csv("EmployeesData.csv")

# Crear subplots con histograma y gráfico de torta
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Histograma
axs[0].hist(df['Education'], bins=10, edgecolor='black')
axs[0].set_title('Histograma de Niveles de Estudio')
axs[0].set_xlabel('Nivel de Estudio')
axs[0].set_ylabel('Frecuencia')
plt.show()


# Cargar el archivo CSV
seguros = pd.read_csv("EmployeesData.csv")
df = pd.read_csv("EmployeesData.csv")

# Crear histograma para la relación entre edad y licencias
plt.figure(figsize=(8, 6))
plt.hist(df[df['Age'] < 30]['LeaveOrNot'], bins=2, edgecolor='black', alpha=0.7, label='Jóvenes')
plt.hist(df[df['Age'] >= 30]['LeaveOrNot'], bins=2, edgecolor='black', alpha=0.7, label='No Jóvenes')
plt.xlabel('Licencias')
plt.ylabel('Frecuencia')
plt.title('Relación entre Edad y Licencias')
plt.legend()
plt.show()

# Cargar el archivo CSV
seguros = pd.read_csv("EmployeesData.csv")

# Contar la cantidad de registros por clase
clase_counts = seguros["PaymentTier"].value_counts()

# Crear la gráfica de barras
plt.figure(figsize=(8, 6))
plt.bar(clase_counts.index, clase_counts.values)
plt.xlabel('Departamento')
plt.ylabel('Cantidad de Empleados')
plt.title('Distribución de Clases en EmployeesData.csv')
plt.show()
