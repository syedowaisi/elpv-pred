from tensorflow.keras import layers, models

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