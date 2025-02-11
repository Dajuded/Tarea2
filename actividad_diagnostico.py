import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import logging
import matplotlib.pyplot as plt

# Ocultar logs innecesarios de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Cargar el dataset (ruta relativa)
dataset_path = './Diagnostico(Datos).csv'
df = pd.read_csv(dataset_path, sep=';', on_bad_lines='skip')

# Eliminar valores nulos
df = df.dropna()

# Convertir texto a números usando LabelEncoder
label_encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() <= 10:
        df[col] = label_encoder.fit_transform(df[col])

# Convertir cadenas a números donde sea posible
df = df.applymap(lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) else x)

# Dividir en 80% entrenamiento y 20% validación
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
val_data = df.iloc[train_size:]

# Seleccionar características y variable objetivo
X_train = train_data.drop(columns=['EVOLUCION'])
y_train = train_data['EVOLUCION']
X_val = val_data.drop(columns=['EVOLUCION'])
y_val = val_data['EVOLUCION']

# Convertir datos a tensores
X_train = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train.values.reshape(-1, 1), dtype=tf.float32)
X_val = tf.convert_to_tensor(X_val.values, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val.values.reshape(-1, 1), dtype=tf.float32)

# Definir el modelo con más capas y activaciones no lineales
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Definir hiperparámetros
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_function = tf.keras.losses.Huber()  # Función de pérdida más robusta

# Compilar el modelo
model.compile(optimizer=optimizer, loss=loss_function, metrics=['mae'])

# Entrenar el modelo
epochs = 300
batch_size = 16
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# Evaluar el modelo
loss, mae = model.evaluate(X_val, y_val)
print(f"Pérdida final en validación: {loss:.4f}, Error Absoluto Medio (MAE): {mae:.4f}")

# Graficar la evolución del loss y MAE
plt.figure(figsize=(12, 5))

# Gráfica del loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss Entrenamiento')
plt.plot(history.history['val_loss'], label='Loss Validación')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.title('Evolución del Loss')

# Gráfica del MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='MAE Entrenamiento')
plt.plot(history.history['val_mae'], label='MAE Validación')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.legend()
plt.title('Evolución del MAE')

plt.show()

# Predicciones
y_pred = model.predict(X_val)

# Gráfico de regresión: Predicciones vs Valores Reales
plt.figure(figsize=(6, 6))
plt.scatter(y_val.numpy(), y_pred, alpha=0.5)
plt.xlabel("Valores Reales (y_val)")
plt.ylabel("Predicciones (y_pred)")
plt.title("Gráfico de Regresión: Predicciones vs. Valores Reales")
plt.plot([min(y_val.numpy()), max(y_val.numpy())], [min(y_val.numpy()), max(y_val.numpy())], color='red', linestyle='dashed')  # Línea ideal
plt.show()
