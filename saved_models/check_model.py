import tensorflow as tf

model = tf.keras.models.load_model("saved_models/resnet18.h5")
print(model.input_shape)