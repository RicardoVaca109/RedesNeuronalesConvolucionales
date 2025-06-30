# -----------------------------------
# PARTE 1 - IMPORTACIÓN DE BIBLIOTECAS
# -----------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------------
# PARTE 2 - CARGA Y PREPROCESAMIENTO
# -----------------------------------
IMG_SIZE = 64
BATCH_SIZE = 32
DATASET_PATH = 'dataset'

train_dir = os.path.join(DATASET_PATH, 'training_set')
test_dir = os.path.join(DATASET_PATH, 'test_set')

# Generador de entrenamiento con data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Generador de validación (solo normalización)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # Dos clases: gato o perro
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # importante para evaluación correcta
)

# -----------------------------------
# PARTE 3 - DEFINIR MODELO CNN
# -----------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Salida binaria
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# -----------------------------------
# PARTE 4 - ENTRENAMIENTO
# -----------------------------------
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=test_generator,
    verbose=1
)

# -----------------------------------
# PARTE 5 - EVALUACIÓN Y VISUALIZACIÓN
# -----------------------------------
# Obtener predicciones
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
y_true = test_generator.classes

# Matriz de confusión y precisión
cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

print("\n📊 Matriz de confusión:")
print(cm)
print(f"✅ Precisión final en test: {acc * 100:.2f}%")

# Gráficas
plt.figure(figsize=(12, 5))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento', marker='o')
plt.plot(history.history['val_accuracy'], label='Validación', marker='s')
plt.title('Precisión del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.grid(True)
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento', marker='o')
plt.plot(history.history['val_loss'], label='Validación', marker='s')
plt.title('Pérdida del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
