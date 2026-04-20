from tensorflow.keras import layers, models

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