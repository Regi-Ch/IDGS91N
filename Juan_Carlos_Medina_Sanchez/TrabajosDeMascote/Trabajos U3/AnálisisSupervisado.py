# Anexo A: Ejemplo en Python con scikit-learn

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Cargar datos
# Si ya tienes un CSV real, usa:
# df = pd.read_csv("ventas_tienda.csv")

# Para ejemplo, generamos un dataset sintético:
np.random.seed(42)
n_meses = 120

df = pd.DataFrame({
    "marketing_gasto": np.random.uniform(10000, 100000, n_meses),
    "visitas_web": np.random.randint(1000, 50000, n_meses),
    "num_promociones": np.random.randint(0, 10, n_meses),
    "precio_promedio": np.random.uniform(100, 1000, n_meses),
    "estacion": np.random.choice(["Primavera", "Verano", "Otono", "Invierno"], n_meses),
})

# Creamos una relación no lineal para las ventas (solo para simular):
ventas_base = 0.5 * df["marketing_gasto"] + 20 * df["num_promociones"]
ruido = np.random.normal(0, 10000, n_meses)
df["ventas"] = ventas_base + ruido

# 2. Separar X e y
X = df.drop("ventas", axis=1)
y = df["ventas"]

# 3. Definir columnas numéricas y categóricas
columnas_numericas = ["marketing_gasto", "visitas_web", "num_promociones", "precio_promedio"]
columnas_categoricas = ["estacion"]

# 4. Transformaciones
preprocesador = ColumnTransformer(
    transformers=[
        ("num", "passthrough", columnas_numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas)
    ]
)

# 5. Definir el modelo
modelo_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

# 6. Crear el pipeline completo
pipeline = Pipeline(steps=[
    ("preprocesador", preprocesador),
    ("modelo", modelo_rf)
])

# 7. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Entrenar el modelo
pipeline.fit(X_train, y_train)

# 9. Realizar predicciones
y_pred = pipeline.predict(X_test)

# 10. Calcular métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:  ", mae)
print("RMSE: ", rmse)
print("R^2:  ", r2)
