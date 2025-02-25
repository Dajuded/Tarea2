import numpy as np
import keras
import keras_tuner as kt
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split

# Función para cargar los datos
def load(f):
    return np.load(f)['arr_0']

# Cargar dataset KMNIST
x_train = load('kmnist-train-imgs.npz')
y_train = load('kmnist-train-labels.npz')
x_test = load('kmnist-test-imgs.npz')
y_test = load('kmnist-test-labels.npz')

# Normalización y preprocesamiento
x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# División en Train (80%) y Validation (20%)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True)

# Función para construir el modelo con hiperparámetros optimizables
def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units_1', min_value=128, max_value=1024, step=128), activation='relu', input_shape=(28*28,)))
    model.add(Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)))
    model.add(Dense(hp.Int('units_2', min_value=128, max_value=512, step=128), activation='relu'))
    model.add(Dropout(hp.Float('dropout_2', 0.2, 0.5, step=0.1)))
    model.add(Dense(10, activation='softmax'))
    
    # Optimización de hiperparámetros adicionales
    optimizerf = hp.Choice('optimizer', ['adam', 'rmsprop'])
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    batch_s = hp.Int('batch_size', min_value=32, max_value=256, step=32)
    
    if optimizerf == 'adam':
        optimizer = Adam(learning_rate=lr)
    else:
        optimizer = RMSprop(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Definir búsqueda de hiperparámetros
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials= 1,  # Número de configuraciones a probar
    directory='kt_tuner',
    project_name='kmnist_mlp_optimized'
)

# Realizar búsqueda de hiperparámetros
tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Obtener mejor conjunto de hiperparámetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Obtener el mejor batch size y número de épocas
batch_s = best_hps.get('batch_size')
num_epochs = 10  # Se puede modificar si es necesario

# Entrenar modelo con los mejores hiperparámetros
history = best_model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_s, validation_data=(x_val, y_val))

# Evaluar en conjunto de prueba
test_loss, test_acc = best_model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Graficar evolución del entrenamiento
plt.figure(figsize=(12, 5))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.show()
