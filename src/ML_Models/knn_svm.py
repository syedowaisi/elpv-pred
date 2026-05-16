import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ------------------------------
# Paths
# ------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

train_dir = os.path.join(BASE_DIR, "preprocessing/processed_data/train")
test_dir = os.path.join(BASE_DIR, "preprocessing/processed_data/test")

IMG_SIZE = (224,224)
BATCH_SIZE = 32

# ------------------------------
# Data Generator
# ------------------------------

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

y_train = train_gen.classes
y_test = test_gen.classes

# ------------------------------
# Feature Extractor (ResNet50)
# ------------------------------

base_model = tf.keras.applications.Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
feature_model = tf.keras.Model(base_model.input, x)

print("Extracting features...")

X_train = feature_model.predict(train_gen, verbose=1)
X_test = feature_model.predict(test_gen, verbose=1)

# ------------------------------
# SVM
# ------------------------------

print("\nTraining SVM...")

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

svm_preds = svm.predict(X_test)

svm_acc = accuracy_score(y_test, svm_preds)

print("SVM Accuracy:", svm_acc * 100)

# ------------------------------
# KNN
# ------------------------------

print("\nTraining KNN...")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_preds = knn.predict(X_test)

knn_acc = accuracy_score(y_test, knn_preds)

print("KNN Accuracy:", knn_acc * 100)