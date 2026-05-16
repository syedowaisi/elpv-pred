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
EPOCHS=5

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






def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image

base_model = tf.keras.applications.ResNet50(
    weights=None,   # 🔥 IMPORTANT (self-supervised)
    include_top=False,
    input_shape=(224,224,3)
)

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128)(x)

encoder = tf.keras.Model(base_model.input, x)

def contrastive_loss(z1, z2):
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)
    return -tf.reduce_mean(tf.reduce_sum(z1 * z2, axis=1))

optimizer = tf.keras.optimizers.Adam(1e-3)


print("\nTraining MoCo-like encoder...\n")

for epoch in range(EPOCHS):

    print(f"Epoch {epoch+1}/{EPOCHS}")

    for step in range(len(train_gen)):

        images, _ = train_gen[step]

        aug1 = augment(images)
        aug2 = augment(images)

        with tf.GradientTape() as tape:
            z1 = encoder(aug1, training=True)
            z2 = encoder(aug2, training=True)

            loss = contrastive_loss(z1, z2)

        grads = tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables))

    print(f"Loss: {loss.numpy():.4f}")


X_train = encoder.predict(train_gen, verbose=1)
X_test = encoder.predict(test_gen,verbose=1)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)
knn_acc = accuracy_score(y_test, preds)

print("KNN Accuracy:", knn_acc * 100)


print("\nTraining SVM...")

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

svm_preds = svm.predict(X_test)

svm_acc = accuracy_score(y_test, svm_preds)

print("SVM Accuracy:", svm_acc * 100)