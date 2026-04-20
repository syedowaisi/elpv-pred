import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------
# Paths
# ------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

test_dir = os.path.join(BASE_DIR, "preprocessing/processed_data/test")
models_dir = os.path.join(BASE_DIR, "..", "saved_models")

# IMG_SIZE = (224,224)
IMG_SIZE = (75,75)
BATCH_SIZE = 32

# ------------------------------
# Load Test Data
# ------------------------------

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(test_generator.class_indices.keys())

print("\nClasses:", test_generator.class_indices)
print("Test samples:", test_generator.samples)

# ------------------------------
# Model List
# ------------------------------

model_files = [
    "resnet18.h5",
    "alexnet.h5",
    "googlenet.h5",
    "xception.h5",
    "cnet.h5",
    "darknet.h5",
    "vit.h5",          # ✅ fixed
    "squeezenet.h5"
]

results = []

print("\nModel Evaluation Results\n")

# ------------------------------
# Evaluate Each Model
# ------------------------------

for model_file in model_files:

    model_path = os.path.join(models_dir, model_file)

    if not os.path.exists(model_path):
        print(f"{model_file} not found")
        continue

    print(f"\nEvaluating {model_file}")

    test_generator.reset()

    model = tf.keras.models.load_model(model_path)

    # Accuracy
    loss, acc = model.evaluate(test_generator, verbose=0)

    # Predictions
    test_generator.reset()
    preds = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_generator.classes

    # Store result
    results.append([model_file.replace(".h5",""), acc])

    print(f"Accuracy: {acc*100:.2f}%")

    # ------------------------------
    # Confusion Matrix
    # ------------------------------
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # ------------------------------
    # Classification Report
    # ------------------------------
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

# ------------------------------
# Final Table
# ------------------------------

print("\nFINAL TEST ACCURACY TABLE\n")

for r in results:
    print(f"{r[0]} → {r[1]*100:.2f}%")