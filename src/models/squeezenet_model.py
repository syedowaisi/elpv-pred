from tensorflow.keras.layers import Conv2D, concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model

def fire_module(x,s1,e1,e3):

    squeeze = Conv2D(s1,(1,1),activation='relu')(x)

    expand1 = Conv2D(e1,(1,1),activation='relu')(squeeze)

    expand3 = Conv2D(e3,(3,3),padding='same',activation='relu')(squeeze)

    return concatenate([expand1,expand3])

def build_squeezenet(input_shape=(64,64,3),num_classes=4):

    inputs = Input(shape=input_shape)

    x = Conv2D(96,(7,7),strides=2,activation='relu')(inputs)

    x = fire_module(x,16,64,64)
    x = fire_module(x,16,64,64)

    x = fire_module(x,32,128,128)

    x = GlobalAveragePooling2D()(x)

    outputs = Dense(num_classes,activation='softmax')(x)

    return Model(inputs,outputs)