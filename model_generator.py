import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
from keras.api.utils import timeseries_dataset_from_array
import tensorflow as tf
import numpy as np
data = pd.read_csv("data/dataset_combinado.csv")
print(data.head())

print("Cantidad de filas y columnas: ", data.shape)
print("Nombre columnas: ", data.columns)
data.info()


# Eliminar registros con valores nulos
data.dropna(inplace=True)



# Agregar características de estacionalidad
# Convertir la columna 'fecha' en datetime
data['fecha'] = pd.to_datetime(data['fecha'])
data['mes'] = data['fecha'].dt.month
data['dia_del_anio'] = data['fecha'].dt.dayofyear

# Normalizar las nuevas características
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['ayacucho', 'caicara', 'ciudad_bolivar', 'palua', 'mes', 'dia_del_anio']])

# Crear secuencias para predecir múltiples días futuros
sequence_length = 90  # Ventana de entrada
forecast_steps = 10   # Número de días futuros a predecir
batch_size = 16       # Reducir el tamaño del batch si es necesario

# Asegurar que los datos estén ordenados por fecha antes de dividir
data.sort_values(by='fecha', inplace=True)

# Convertir a array después de escalar
data_array = scaler.fit_transform(data[['ayacucho', 'caicara', 'ciudad_bolivar', 'palua', 'mes', 'dia_del_anio']])

# Elegir índice de corte para training y testing (por ejemplo 80% para training)
split_index = int(len(data_array) * 0.8)

# Asegurarse de que haya suficientes datos para generar secuencias
if split_index < sequence_length + forecast_steps:
    raise ValueError("El conjunto de datos es demasiado pequeño para generar las secuencias requeridas.")

train_data = data_array[:split_index]
test_data = data_array[split_index:]



# Crear secuencias para predecir múltiples días futuros
train_dataset = timeseries_dataset_from_array(
    data=train_data,
    targets=np.array([train_data[i:i + forecast_steps, :4] for i in range(sequence_length, len(train_data) - forecast_steps)]),
    sequence_length=sequence_length,
    batch_size=batch_size
)

test_dataset = timeseries_dataset_from_array(
    data=test_data,
    targets=np.array([test_data[i:i + forecast_steps, :4] for i in range(sequence_length, len(test_data) - forecast_steps)]),
    sequence_length=sequence_length,
    batch_size=batch_size
)

# Ajustar el modelo para predecir forecast_steps valores futuros
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(sequence_length, train_data.shape[1])),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),  # Capturar patrones locales
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(forecast_steps * 4),  # Ajustar el número de unidades para que coincida con forecast_steps * 4
    tf.keras.layers.Reshape((forecast_steps, 4))  # Cambiar la forma de la salida a (forecast_steps, 4)
])
# Compilar el modelo
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), 
    loss="mean_squared_error", 
    metrics=["mean_absolute_error"]
)

# Entrenar el modelo con más épocas
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5,  # Incrementar el número de épocas
    batch_size=32,  # Ajustar el tamaño del lote
    verbose=1
)

# Graficar la pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.show()

# Tomar la última secuencia del set de test para predecir los próximos 10 días
last_sequence = test_data[-sequence_length:]
last_sequence = last_sequence.reshape((1, sequence_length, test_data.shape[1]))

# Predecir los próximos 10 días para las 4 ciudades
future_pred = model.predict(last_sequence)  # shape: (1, 10, 4)
future_pred = future_pred[0]  # shape: (10, 4)

# Desnormalizar la salida para obtener los valores reales
future_pred_real = scaler.inverse_transform(
    np.hstack([future_pred, np.zeros((forecast_steps, 2))])  # Agregar columnas ficticias para desnormalizar
)[:, :4]  # Tomar solo las primeras 4 columnas

# Obtener los valores reales correspondientes (si están disponibles)
try:
    real_future = scaler.inverse_transform(
        np.hstack([test_data[-sequence_length: -sequence_length + forecast_steps, :4], np.zeros((forecast_steps, 2))])
    )[:, :4]
except Exception:
    real_future = None

print("Predicciones (10 días x 4 ciudades):")
print(future_pred_real)

if real_future is not None and real_future.shape == (10, 4):
    print("\nValores reales (si están disponibles):")
    print(real_future)
else:
    print("\nNo hay valores reales disponibles para los próximos 10 días.")

# Graficar
ciudades = ['ayacucho', 'caicara', 'ciudad_bolivar', 'palua']
dias = np.arange(1, forecast_steps + 1)

plt.figure(figsize=(10, 6))
for i, ciudad in enumerate(ciudades):
    plt.figure(figsize=(6, 4))
    plt.plot(dias, future_pred_real[:, i], marker='o', label=f'Predicción - {ciudad}')
    if real_future is not None and real_future.shape == (forecast_steps, 4):
        plt.plot(dias, real_future[:, i], marker='x', linestyle='--', label=f'Real - {ciudad}')
    plt.xlabel('Día futuro')
    plt.ylabel('Valor real')
    plt.title(f'Predicción vs Real - {ciudad}')
    plt.legend()
    plt.grid()
    plt.show()

