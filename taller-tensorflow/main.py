
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ==========================================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================================

SEMILLA = 42
#TASA_DE_APRENDIZAJE = 0.001
#TASA_DE_APRENDIZAJE = 0.5
TASA_DE_APRENDIZAJE = 0.00001
TAMANO_LOTE = 64
EPOCAS = 10

# Identificador para organizar los resultados.
EXPERIMENTO = "experimento_3_lr_bajo"

tf.keras.utils.set_random_seed(SEMILLA)
np.random.seed(SEMILLA)

print(f"TensorFlow: {tf.__version__}")

# ==========================================================
# 2. CARGA Y PREPARACIÓN DE MNIST
# ==========================================================

mnist = tf.keras.datasets.mnist

(X_train_raw, y_train), (X_test_raw, y_test) = (
    mnist.load_data()
)

# Normalizamos los píxeles al rango [0, 1].
X_train = X_train_raw.astype("float32") / 255.0
X_test = X_test_raw.astype("float32") / 255.0

print("Forma de entrenamiento:", X_train.shape)
print("Forma de prueba:", X_test.shape)

# ==========================================================
# 3. CONSTRUCCIÓN DEL MLP BASE
# ==========================================================


def construir_mlp():
    modelo = models.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Flatten(name="Capa_Entrada_Flatten"),
            layers.Dense(
                128, activation="relu",
                name="Capa_Oculta_1"
            ),
            layers.Dense(
                64, activation="relu",
                name="Capa_Oculta_2"
            ),
            layers.Dense(
                10, activation="softmax",
                name="Capa_Salida_Softmax"
            ),
        ],
        name="MLP_UNINTEP_RA2",
    )
    return modelo


modelo_mlp = construir_mlp()
modelo_mlp.summary()

# ==========================================================
# 4. COMPILACIÓN
# ==========================================================

optimizador = optimizers.Adam(
    learning_rate=TASA_DE_APRENDIZAJE
)

modelo_mlp.compile(
    optimizer=optimizador,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ==========================================================
# 5. ENTRENAMIENTO
# ==========================================================

print("\nIniciando entrenamiento...")

inicio = time.perf_counter()

historial = modelo_mlp.fit(
    X_train,
    y_train,
    epochs=EPOCAS,
    batch_size=TAMANO_LOTE,
    validation_split=0.2,
    verbose=1,
)

tiempo_entrenamiento = time.perf_counter() - inicio

# ==========================================================
# 6. EVALUACIÓN FINAL
# ==========================================================

loss_test, accuracy_test = modelo_mlp.evaluate(
    X_test, y_test, verbose=0
)

loss_train_final = historial.history["loss"][-1]
loss_val_final = historial.history["val_loss"][-1]
accuracy_train_final = historial.history["accuracy"][-1]
accuracy_val_final = historial.history["val_accuracy"][-1]

print("\n========== RESULTADOS ==========")
print(f"Experimento: {EXPERIMENTO}")
print(f"Learning rate: {TASA_DE_APRENDIZAJE}")
print(f"Loss de entrenamiento: {loss_train_final:.4f}")
print(f"Loss de validación: {loss_val_final:.4f}")
print(f"Loss de prueba: {loss_test:.4f}")
print(f"Accuracy de entrenamiento: {accuracy_train_final:.4f}")
print(f"Accuracy de validación: {accuracy_val_final:.4f}")
print(f"Accuracy de prueba: {accuracy_test:.4f}")
print(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} segundos")
print("================================")

# ==========================================================
# 7. GRÁFICAS DE PÉRDIDA Y PRECISIÓN
# ==========================================================

epocas = range(1, EPOCAS + 1)

plt.figure(figsize=(9, 5))
plt.plot(epocas, historial.history["loss"],
         marker="o", label="Entrenamiento")
plt.plot(epocas, historial.history["val_loss"],
         marker="o", label="Validación")
plt.title(f"Pérdida por época - {EXPERIMENTO}")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"loss_{EXPERIMENTO}.png", dpi=150)
plt.show()
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(epocas, historial.history["accuracy"],
         marker="o", label="Entrenamiento")
plt.plot(epocas, historial.history["val_accuracy"],
         marker="o", label="Validación")
plt.title(f"Precisión por época - {EXPERIMENTO}")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"accuracy_{EXPERIMENTO}.png", dpi=150)
plt.show()
plt.close()