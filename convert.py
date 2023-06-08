import tensorflow as tf
import tensorflow_model_optimization as tfmot

model = tf.keras.models.load_model('C:\\Users\\Gayathri\\Documents\\Final Year Project\\Tomato Leaf disease detection\\model_tomato_inception.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_model_fp16_file = "Model.tflite"
open(tflite_model_fp16_file, "wb").write(tflite_model)

