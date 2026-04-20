import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model

def build_vit(input_shape=(64,64,3),num_classes=4):

    inputs = Input(shape=input_shape)

    x = Flatten()(inputs)

    x = Dense(512,activation='relu')(x)
    x = Dense(256,activation='relu')(x)

    outputs = Dense(num_classes,activation='softmax')(x)

    return Model(inputs,outputs)