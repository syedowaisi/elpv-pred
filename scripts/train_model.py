import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, Xception, InceptionV3, MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Conv2D, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler6
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ================= PATHS ================= #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "split_data", "train")
VAL_DIR   = os.path.join(BASE_DIR, "dataset", "split_data", "val")
TEST_DIR  = os.path.join(BASE_DIR, "dataset", "split_data", "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ================= SETTINGS ================= #
IMG_SIZE = (224,224)
BATCH_SIZE = 25
EPOCHS = 40
LR = 1e-4

BINARY_MODE = False   # 🔁 True = binary classification
NUM_CLASSES = 2 if BINARY_MODE else 4

# ================= COSINE LR ================= #
def cosine_lr(epoch):
    return LR * 0.5 * (1 + np.cos(np.pi * epoch / EPOCHS))

lr_scheduler = LearningRateScheduler(cosine_lr)

# ================= DATA ================= #
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical" if not BINARY_MODE else "binary")

val_gen = datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical" if not BINARY_MODE else "binary")

test_gen = datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical" if not BINARY_MODE else "binary",
    shuffle=False)

# ================= SE BLOCK ================= #
def se_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters//ratio, activation="relu")(se)
    se = Dense(filters, activation="sigmoid")(se)
    return Multiply()([input_tensor, se])

# ================= MODEL BUILDER ================= #
def build_model(base):
    base.trainable = False
    x = base.output
    x = se_block(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(NUM_CLASSES, activation="softmax" if not BINARY_MODE else "sigmoid")(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=Adam(LR),
        loss="categorical_crossentropy" if not BINARY_MODE else "binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ================= ViT ================= #
def build_vit():
    from tensorflow.keras.applications import EfficientNetB0
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))
    return build_model(base)

# ================= MODELS ================= #
models = {
    "resnet": build_model(ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))),
    "xception": build_model(Xception(weights="imagenet", include_top=False, input_shape=(224,224,3))),
    "inception": build_model(InceptionV3(weights="imagenet", include_top=False, input_shape=(224,224,3))),
    "mobilenet": build_model(MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))),
    "vit": build_vit()
}

histories = {}
predictions = []

# ================= TRAIN (BAGGING) ================= #
for name, model in models.items():
    print(f"\nTraining {name.upper()}...\n")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[lr_scheduler]
    )

    model.save(os.path.join(MODEL_DIR, f"{name}.keras"))
    histories[name] = history

    preds = model.predict(test_gen)
    predictions.append(preds)

# ================= SOFT VOTING ================= #
ensemble_preds = np.mean(predictions, axis=0)
y_true = test_gen.classes
y_pred = np.argmax(ensemble_preds, axis=1)

acc = np.mean(y_pred == y_true)
print("\nENSEMBLE ACCURACY:", acc)

# ================= CONFUSION MATRIX ================= #
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))

# ================= PLOTS ================= #
for name, history in histories.items():
    plt.figure()
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.legend()
    plt.title(name)
    plt.savefig(os.path.join(RESULT_DIR, f"{name}_accuracy.png"))

    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.legend()
    plt.title(name)
    plt.savefig(os.path.join(RESULT_DIR, f"{name}_loss.png"))

print("\nAll models trained. Results saved in /results")