from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.models import Model

def inception_module(x, f1, f3, f5):

    conv1 = Conv2D(f1, (1,1), activation='relu', padding='same')(x)

    conv3 = Conv2D(f3, (3,3), activation='relu', padding='same')(x)

    conv5 = Conv2D(f5, (5,5), activation='relu', padding='same')(x)

    pool = MaxPooling2D((3,3), strides=1, padding='same')(x)

    return concatenate([conv1, conv3, conv5, pool])


def build_googlenet(input_shape=(64,64,3), num_classes=4):

    inputs = Input(shape=input_shape)

    # 🔥 Better start for small images
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)   # 64 → 32

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Inception blocks
    x = inception_module(x, 32, 64, 16)
    x = MaxPooling2D((2,2))(x)   # 32 → 16

    x = inception_module(x, 64, 96, 32)
    x = MaxPooling2D((2,2))(x)   # 16 → 8

    # Global pooling
    x = GlobalAveragePooling2D()(x)

    # Dense
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)   # 🔥 important

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)