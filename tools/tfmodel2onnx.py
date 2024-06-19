import tensorflow as tf
import os
os.environ['TF_KERAS'] = '1'
import onnxmltools


model = tf.keras.models.load_model('models/ASMNet_300W_ASMLoss.h5')
onnx_model = onnxmltools.convert_keras(model) 

onnxmltools.utils.save_model(onnx_model, 'models/asmnet.onnx')