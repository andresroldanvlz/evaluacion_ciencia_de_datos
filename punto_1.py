import pandas as pd
import numpy as np

# Cargar el archivo CSV (reemplaza 'ruta_del_archivo.csv' con la ubicación real del archivo)
ruta_del_archivo = 'datos_procesados.csv'
df = pd.read_csv(ruta_del_archivo)

# Verificar valores faltantes
print("Valores faltantes por columna:")
print(df.isnull().sum())

# Convertir la columna 'LeaveOrNot' a etiquetas categóricas
df['LeaveOrNot'] = df['LeaveOrNot'].map({0: 'Not Leave', 1: 'Leave'})

# Eliminar filas con valores faltantes en 'ExperienceInCurrentDomain' y 'JoiningYear'
df.dropna(subset=['ExperienceInCurrentDomain', 'JoiningYear'], inplace=True)

# Imputar datos faltantes en la columna 'Age' con la media
mean_age = df['Age'].mean()
df['Age'].fillna(mean_age, inplace=True)

# Imputar datos faltantes en la columna 'PaymentTier' con la moda
mode_payment_tier = df['PaymentTier'].mode().iloc[0]
df['PaymentTier'].fillna(mode_payment_tier, inplace=True)

# Eliminar registros con valores atípicos basados en IQR (excepto 'LeaveOrNot')
numeric_columns = df.select_dtypes(include=np.number).columns
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Guardar el DataFrame procesado en un nuevo archivo CSV
df.to_csv('', index=False)

print("Procesamiento completado. El archivo 'datos_procesados.csv' contiene los datos procesados.")
