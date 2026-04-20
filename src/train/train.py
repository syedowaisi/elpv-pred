import sys
import os
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ------------------------------
# Paths
# ------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

train_dir = os.path.join(BASE_DIR, "preprocessing/processed_data/train")
val_dir = os.path.join(BASE_DIR, "preprocessing/processed_data/val")

IMG_SIZE = (75,75)   # ✅ FIXED// for xception 
# IMG_SIZE= (64,64)  #for all remaining models 
BATCH_SIZE = 32
EPOCHS = 20          # ✅ FIXED

print("Train path:", train_dir)
print("Path exists:", os.path.exists(train_dir))

# ------------------------------
# Data Generators
# ------------------------------

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=15,
#     zoom_range=0.1,
#     horizontal_flip=True
# )
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
class_weight = {
    0:1.0,
    1:2.0,   # mild boosted
    2:1.2,
    3:1.0
}
# ------------------------------
# Import models
# ------------------------------
from tensorflow.keras import layers, models

def alexnet_64():

    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(96, (5,5), padding='same', activation='relu', input_shape=(64,64,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    # Block 2
    model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    # Block 3
    model.add(layers.Conv2D(384, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(384, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    # Global pooling
    model.add(layers.GlobalAveragePooling2D())

    # Dense layers
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))   # 🔥 important

    model.add(layers.Dense(4, activation='softmax'))

    return model


def cnet_64():

    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(64,64,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))   # 64 → 32

    # Block 2
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))   # 32 → 16

    # Block 3
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))   # 16 → 8

    # Global pooling
    model.add(layers.GlobalAveragePooling2D())

    # Dense
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))   # 🔥 important

    model.add(layers.Dense(4, activation='softmax'))

    return model

def build_darknet(input_shape=(64,64,3), num_classes=4):

    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    # Block 2
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    # Block 3
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    # Block 4
    model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())

    # Global pooling
    model.add(layers.GlobalAveragePooling2D())

    # Dense
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))   # 🔥 important

    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_xception(input_shape=(75,75,3), num_classes=4):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # 🔥 Freeze most layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 🔥 Add dropout
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    return model

from tensorflow.keras import layers, Model

def build_vit(input_shape=(64,64,3), num_classes=4):

    inputs = layers.Input(shape=input_shape)

    # 🔹 Patch creation
    patches = layers.Conv2D(64, kernel_size=8, strides=8)(inputs)  # 64→8 patches
    patches = layers.Reshape((-1, 64))(patches)

    # 🔹 Positional encoding
    positions = tf.range(start=0, limit=patches.shape[1], delta=1)
    pos_embed = layers.Embedding(input_dim=patches.shape[1], output_dim=64)(positions)

    x = patches + pos_embed

    # 🔹 Transformer block
    for _ in range(2):
        x1 = layers.LayerNormalization()(x)
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x1, x1)
        x2 = layers.Add()([x, attention])

        x3 = layers.LayerNormalization()(x2)
        mlp = layers.Dense(128, activation='relu')(x3)
        mlp = layers.Dense(64)(mlp)

        x = layers.Add()([x2, mlp])

    # 🔹 Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

# from models.resnet18_model import build_resnet18
# from models.alexnet_model import alexnet_64
# from models.googlenet_model import build_googlenet
# from models.xception_model import build_xception
# from models.cnet_model import cnet_64
# from models.darknet_model import build_darknet
# from models.vit_model import build_vit
# from models.squeezenet_model import build_squeezenet

# ------------------------------
# Callbacks
# ------------------------------

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6
)

# ------------------------------
# Training function
# ------------------------------

# def train_model(model, model_name):

#     tf.keras.backend.clear_session()

#     print(f"\nTraining {model_name}...\n")

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # ✅ FIXED
#         loss="categorical_crossentropy",
#         metrics=["accuracy"]
#     )

#     history = model.fit(
#         train_generator,
#         validation_data=val_generator,
#         epochs=EPOCHS,
#         callbacks=[early_stop, reduce_lr]   # ✅ IMPORTANT
#     )

#     os.makedirs("saved_models", exist_ok=True)
#     model.save(f"saved_models/{model_name}.h5")

#     train_acc = history.history['accuracy'][-1]
#     val_acc = history.history['val_accuracy'][-1]

#     return train_acc, val_acc


# below training function just for xception since finetuning last layers
def train_model(model, model_name):

    tf.keras.backend.clear_session()

    print(f"\nTraining {model_name} (Stage 1 - Frozen)...\n")

    # ------------------------------
    # STAGE 1 → Freeze base model
    # ------------------------------
    for layer in model.layers:
        layer.trainable = False

    # Keep last Dense layers trainable
    for layer in model.layers[-5:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history1 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=5,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight
    )

    # ------------------------------
    # STAGE 2 → Fine-tuning
    # ------------------------------
    print(f"\nFine-tuning {model_name} (Stage 2)...\n")

    for layer in model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # 🔥 lower LR
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight
    )

    # ------------------------------
    # Save model
    # ------------------------------
    os.makedirs("saved_models", exist_ok=True)
    model.save(f"saved_models/{model_name}.h5")

    # ------------------------------
    # Final Accuracy
    # ------------------------------
    train_acc = history2.history['accuracy'][-1]
    val_acc = history2.history['val_accuracy'][-1]

    return train_acc, val_acc

# ------------------------------
# Train models
# ------------------------------

models_list = [
    ("xception", build_xception),
    # ("resnet18", build_resnet18),
    # ("googlenet", build_googlenet),
    # ("squeezenet", build_squeezenet),
    # ("alexnet", alexnet_64),   # optional
    # ("cnet", cnet_64),
    # ("darknet", build_darknet),
    # ("vit", build_vit)         # optional
]

results = []

print("\nTraining starting...\n")

for name, fn in models_list:
    try:
        train_acc, val_acc = train_model(fn(), name)
        results.append([name, train_acc, val_acc])
    except Exception as e:
        print(f"Error in {name}: {e}")

# ------------------------------
# Results
# ------------------------------

print("\nFINAL TRAINING RESULTS:\n")

for r in results:
    print(f"{r[0]} → Train: {r[1]*100:.2f}% | Val: {r[2]*100:.2f}%")