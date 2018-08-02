
# coding: utf-8

# In[7]:

from __future__ import absolute_import
from __future__ import print_function

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import numpy as np
import pandas as pd
import urllib
from sklearn.cross_validation import train_test_split
import glob
import os
import pickle

from trainer.environment import create_trainer_environment

# the trainer environment contains useful information about
env = create_trainer_environment()
print('creating SageMaker trainer environment:\n%s' % str(env))

## LOAD the data from train and test channels
npX_keras = pickle.load(open(os.path.join(env.channel_dirs['train'], 'npX_keras.pkl'), "rb"))
oh_npY = pickle.load(open(os.path.join(env.channel_dirs['train'], 'oh_npY.pkl'), "rb"))

#Lest split the data into train and validation
train_X, validation_X, train_y, validation_y = train_test_split(npX_keras, oh_npY, test_size=0.2, random_state=1001)

batch_size = 32
epochs = 1

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (128, 128, 3))


# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:5]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(12, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)



# In[57]:


# Initiate the train and test generators with data Augumentation 
img_width, img_height = 128, 128

train_datagen = ImageDataGenerator(
                    rescale = 1./255,
                    horizontal_flip = True,
                    fill_mode = "nearest",
                    zoom_range = 0.3,
                    width_shift_range = 0.3,
                    height_shift_range=0.3,
                    rotation_range=30)

test_datagen = ImageDataGenerator(
                    rescale = 1./255,
                    horizontal_flip = True,
                    fill_mode = "nearest",
                    zoom_range = 0.3,
                    width_shift_range = 0.3,
                    height_shift_range=0.3,
                    rotation_range=30)

train_generator = train_datagen.flow(train_X, train_y,
                    batch_size=batch_size)

validation_generator = test_datagen.flow(validation_X, validation_y)

# Save the model according to the conditions  
#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, 
#                             save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')


# ## PREP for AWS Sagemaker - start.py

# In[30]:


MODEL_NAME = 'seedling_model.h5'

# getting the hyperparameters
batch_size = env.hyperparameters.get('batch_size', object_type=int)
learning_rate = env.hyperparameters.get('learning_rate', default=.0001, object_type=float)
EPOCHS = env.hyperparameters.get('epochs', default=10, object_type=int)


# TRAIN Model
n_train_samples = train_X.shape[0]
n_validation_samples = validation_X.shape[0] *1.0

# compile the model 
#model_final.compile(loss="categorical_crossentropy", 
#                    optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

model_final.compile(loss="categorical_crossentropy", 
                    optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9), metrics=["accuracy"])

model_final.fit_generator(
                    train_generator,
                    samples_per_epoch=n_train_samples/batch_size,
                    epochs=EPOCHS,
                    validation_data=validation_generator,
                    validation_steps=n_validation_samples/batch_size)

# Save model and weights
model_path = os.path.join(env.model_dir, MODEL_NAME)
model_final.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model_final.evaluate_generator(validation_generator, n_validation_samples/batch_size, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

