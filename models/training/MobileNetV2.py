#region Imports
# Importieren der Bibliotheken
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Rescaling,RandomFlip,RandomBrightness,RandomRotation
from datetime import datetime
#endregion


#region constants
num_classes = 127
#enregion

#region Import Pictures
batch_size = 20
img_height = 256
img_width = 256
img_shape = (img_width,img_height,3)
input_shape=(None,)+img_shape

Dataset = 'path_to_preprocessed_dataset'

traindir = './'+Dataset+'/train/'
valdir   = './'+Dataset+'/val/'
testdir  = './'+Dataset+'/test/'

train_ds = tf.keras.utils.image_dataset_from_directory(
  traindir,
  label_mode = 'categorical',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  interpolation='lanczos5')

val_ds   = tf.keras.utils.image_dataset_from_directory(
  valdir,
  label_mode = 'categorical',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  interpolation='lanczos5')

test_ds   = tf.keras.utils.image_dataset_from_directory(
  testdir,
  label_mode = 'categorical',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  interpolation='lanczos5')

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds   =   val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  =  test_ds.cache().prefetch(buffer_size=AUTOTUNE)
#endregion


#region Preprocessing
rescale = tf.keras.Sequential([
  layers.Rescaling(1./127.5,offset=-1)
])


data_augmentation = tf.keras.Sequential([
  RandomFlip('horizontal'),
  RandomFlip('vertical'),
  RandomBrightness(0.2), 
])
#endregion


#region Model
model_name="MobileNETV2"+'_'+Dataset 

base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                               include_top=False,
                                               weights='imagenet')


base_model.trainable = False
model = Sequential()

model.add(data_augmentation)
model.add(rescale)
model.add(base_model)

model.add(GlobalAveragePooling2D())
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


model.build(input_shape)
#endregion


#region Training-Options
model_initial_epochs=50
model_initial_patience=5
model_patience=50
model_finetune_at= 140
model_finetune_epochs = 1200
model_optimizer='adam'

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer = model_optimizer)
model.summary()

date = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_bestmodel = './checkpoints/checkpoint_bestmodel_'+model_name+'_'+date+'.weights.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_bestmodel,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=model_initial_patience)
#endregion

#region Training
history_initial = model.fit( 
    train_ds, 
    epochs=model_initial_epochs,
    callbacks=[model_checkpoint_callback,early_callback],
    validation_data = val_ds)

base_model.trainable = True
for layer in base_model.layers[:model_finetune_at]:
  layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=model_optimizer,
              metrics=['accuracy'])

model.summary()

early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=model_patience)

history_finetune = model.fit( 
    train_ds, 
    epochs=model_finetune_epochs,
    callbacks=[model_checkpoint_callback,early_callback],
    validation_data = val_ds)
#endregion


#region Save Models

model.save_weights('./checkpoints/checkpoint_lastmodel_'+model_name+'_'+date+'.weights.h5')
model.save('./model_storage/last_model_'+model_name+'_'+date+'.keras')
model.load_weights('./checkpoints/checkpoint_bestmodel_'+model_name+'_'+date+'.weights.h5')
model.save('./model_storage/best_model_'+model_name+'_'+date+'.keras')
#endregion

#Ermittlung der Erkennungsrate mit den Testdaten
print('Best model:')
metrics = model.evaluate(x=test_ds)
accuracy=metrics[1]
Name = model_name+'_'
Accuracy=f"{accuracy*100:.4f}%_"
Epochs=f"{model_initial_epochs+model_finetune_epochs:d}_Epochs_"
Optimizer="Optimizer_"+model_optimizer+"_"
Patience=f"{model_patience:d}_Patience_"
formatted_name=Accuracy+Name+Epochs+Optimizer+Patience

history_initial = history_initial.history
history_finetune = history_finetune.history

hist_initial_df = pd.DataFrame(history_initial)
hist_finetune_df = pd.DataFrame(history_finetune)

hist_csv_file = './history/'+formatted_name+'_initial'+'.csv'
with open (hist_csv_file, mode='w') as f:
    hist_initial_df.to_csv(f)
hist_csv_file = './history/'+formatted_name+'_finetune'+'.csv'
with open (hist_csv_file, mode='w') as f:
    hist_finetune_df.to_csv(f)


np.save('./history/'+formatted_name+'_inital.npy',history_initial)
np.save('./history/'+formatted_name+'_finetune.npy',history_finetune)


model.save('./best_models/best_model_'+formatted_name+'_'+date+'.keras')
model.save_weights('./best_weights/best_model_'+formatted_name+'_'+date+'.weights.h5')
