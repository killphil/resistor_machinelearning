import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Average, GlobalAveragePooling2D, Dense, Activation, Dropout,Rescaling,RandomFlip,RandomBrightness,RandomRotation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

num_classes = 127
batch_size = 20
img_height = 256
img_width = 256
img_shape = (img_width,img_height,3)
input_shape=(None,)+img_shape

#region Custom-Layers
rescale = tf.keras.Sequential([
  layers.Rescaling(1./127.5,offset=-1)
])

data_augmentation = tf.keras.Sequential([
  RandomFlip('horizontal'),
  RandomFlip('vertical'),
  RandomBrightness(0.2), 
])
#emdregion

#region MobileNetV2
MobileNetV2_base = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                               include_top=False,
                                               weights='imagenet')

MobileNetV2 = Sequential()
MobileNetV2.add(data_augmentation)
MobileNetV2.add(rescale)
MobileNetV2.add(MobileNetV2_base)
MobileNetV2.add(GlobalAveragePooling2D())
MobileNetV2.add(Dense(1000))
MobileNetV2.add(Activation('relu'))
MobileNetV2.add(Dropout(0.5))
MobileNetV2.add(Dense(250))
MobileNetV2.add(Activation('relu'))
MobileNetV2.add(Dropout(0.5))
MobileNetV2.add(Dense(100))
MobileNetV2.add(Activation('relu'))
MobileNetV2.add(Dropout(0.5))
MobileNetV2.add(Dense(num_classes))
MobileNetV2.add(Activation('softmax'))
MobileNetV2.build(input_shape)
MobileNetV2.compile(loss='categorical_crossentropy',
              metrics=['accuracy',tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
MobileNetV2_trained = MobileNetV2
MobileNetV2_trained.load_weights('./models/weights/26.7717%_MobileNETV2_Output_Dataset.weights.h5')
#endregion

#region EfficientNetV2S
EfficientNetV2S_base = tf.keras.applications.EfficientNetV2S(input_shape=img_shape,
                                               include_top=False,
                                               weights='imagenet',
                                               include_preprocessing = True)

EfficientNetV2S = Sequential()
EfficientNetV2S.add(data_augmentation)
EfficientNetV2S.add(EfficientNetV2S_base)
EfficientNetV2S.add(GlobalAveragePooling2D())
EfficientNetV2S.add(Dense(1000))
EfficientNetV2S.add(Activation('relu'))
EfficientNetV2S.add(Dropout(0.6))
EfficientNetV2S.add(Dense(250))
EfficientNetV2S.add(Activation('relu'))
EfficientNetV2S.add(Dropout(0.6))
EfficientNetV2S.add(Dense(100))
EfficientNetV2S.add(Activation('relu'))
EfficientNetV2S.add(Dropout(0.6))
EfficientNetV2S.add(Dense(num_classes))
EfficientNetV2S.add(Activation('softmax'))
EfficientNetV2S.build(input_shape)
EfficientNetV2S.compile(loss='categorical_crossentropy',
              metrics=['accuracy',tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
EfficientNetV2S_trained = EfficientNetV2S
EfficientNetV2S_trained.load_weights('./models/weights/77.1654%_EfficientNetV2S.weights.h5')
#endregion

#region ConvNeXtTiny
ConvNeXtTiny_base = tf.keras.applications.ConvNeXtTiny(input_shape=img_shape,
                                               include_top=False,
                                               weights='imagenet',
                                               include_preprocessing = True)
ConvNeXtTiny = Sequential()
ConvNeXtTiny.add(data_augmentation)
ConvNeXtTiny.add(ConvNeXtTiny_base)
ConvNeXtTiny.add(GlobalAveragePooling2D())
ConvNeXtTiny.add(Dense(1000))
ConvNeXtTiny.add(Activation('relu'))
ConvNeXtTiny.add(Dropout(0.6))
ConvNeXtTiny.add(Dense(250))
ConvNeXtTiny.add(Activation('relu'))
ConvNeXtTiny.add(Dropout(0.6))
ConvNeXtTiny.add(Dense(100))
ConvNeXtTiny.add(Activation('relu'))
ConvNeXtTiny.add(Dropout(0.6))
ConvNeXtTiny.add(Dense(num_classes))
ConvNeXtTiny.add(Activation('softmax'))
ConvNeXtTiny.build(input_shape)
ConvNeXtTiny.build(input_shape)
ConvNeXtTiny.compile(loss='categorical_crossentropy',
              metrics=['accuracy',tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
ConvNeXtTiny_trained = ConvNeXtTiny
ConvNeXtTiny_trained.load_weights('./models/weights/70.8661%_ConvNextTiny.weights.h5')
#endregion


#region EnsembleModel
ensemble_base_models = [ConvNeXtTiny_trained, EfficientNetV2S_trained]
ensemble_model_input = tf.keras.Input(shape=img_shape)
ensemble_base_outputs = [model(ensemble_model_input) for model in ensemble_base_models]
ensemble_output = tf.keras.layers.Average()(ensemble_base_outputs)
Ensemble_model_trained = tf.keras.Model(inputs=ensemble_model_input, outputs=ensemble_output)
Ensemble_model_trained.compile(loss='categorical_crossentropy',
              metrics=['accuracy',tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
#endregion

