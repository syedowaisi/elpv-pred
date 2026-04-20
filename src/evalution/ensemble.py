import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

# ------------------------------
# Paths
# ------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

test_dir = os.path.join(BASE_DIR, "preprocessing/processed_data/test")
models_dir = os.path.join(BASE_DIR, "..", "saved_models")

# IMG_SIZE = (224,224)
IMG_SIZE = (64,64)

BATCH_SIZE = 32

# ------------------------------
# Load test data
# ------------------------------

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

y_true = test_generator.classes

# ------------------------------
# Load selected models
# ------------------------------

model_names = ["xception.h5", "vit.h5", "darknet.h5", "googlenet.h5", "cnet.h5"]

models = []

for name in model_names:
    path = os.path.join(models_dir, name)
    model = tf.keras.models.load_model(path)
    models.append(model)

# ------------------------------
# Ensemble Prediction
# ------------------------------

predictions = []

for model in models:
    test_generator.reset()
    preds = model.predict(test_generator, verbose=0)
    predictions.append(preds)

# Average predictions (soft voting)
avg_preds = np.mean(predictions, axis=0)

y_pred = np.argmax(avg_preds, axis=1)

# ------------------------------
# Accuracy
# ------------------------------

acc = accuracy_score(y_true, y_pred)

print("\n🔥 ENSEMBLE ACCURACY:", acc*100)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class_names = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Ensemble Model")

plt.show()