import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.api.utils import timeseries_dataset_from_array


data = pd.read_csv("data/dataset_combinado.csv")
print(data.head())

print("Cantidad de filas y columnas: ", data.shape)
print("Nombre columnas: ", data.columns)
data.info()


# Normalizar los datos
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.iloc[:, 1:]) # Excluye la columna fecha

# Dividir los datos en entrenamiento y prueba
train_data, test_data = train_test_split(data_scaled, test_size=0.2, random_state=42)

# Crear secuencias para predecir múltiples días futuros
sequence_length = 30  # Ventana de entrada
forecast_steps = 5    # Número de días futuros a predecir

train_dataset = timeseries_dataset_from_array(
    data=train_data,
    targets=train_data[sequence_length:sequence_length + forecast_steps],
    sequence_length=sequence_length,
    batch_size=32
)

test_dataset = timeseries_dataset_from_array(
    data=test_data,
    targets=test_data[sequence_length:sequence_length + forecast_steps],
    sequence_length=sequence_length,
    batch_size=32
)

# Imprimir un ejemplo de batch del conjunto de entrenamiento
for batch in train_dataset.take(1):
    inputs, targets = batch
    print("Train Inputs shape:", inputs.shape)
    print("Train Targets shape:", targets.shape)

# Imprimir un ejemplo de batch del conjunto de prueba
for batch in test_dataset.take(1):
    inputs, targets = batch
    print("Test Inputs shape:", inputs.shape)
    print("Test Targets shape:", targets.shape)

# Imprimir los primeros valores del conjunto de entrenamiento
print("Primeros valores de train_data:")
print(train_data[:5])  # Muestra los primeros 5 registros

