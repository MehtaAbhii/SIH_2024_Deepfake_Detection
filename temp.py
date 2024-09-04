import tensorflow as tf


config = "model_config.json"
weights = "model_weights.h5"
model = tf.keras.models.model_from_json(open(config).read())
model.load_weights(weights)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)