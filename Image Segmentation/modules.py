import os
import numpy as np
from tqdm import tqdm
import imageio as iio
import cv2 as cv
import pickle
from natsort import natsorted
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from PIL import Image
from keras import backend as K

from google.colab import drive
drive.mount('/content/drive')

def segmentation_model(
  data_folder = data_folder_i
):
  os.chdir(data_folder)

  with open('image_segementation_data_final_og.pkl', 'rb') as handle:
      image_segementation_stuff = pickle.load(handle)

  pic_train = image_segementation_stuff[0]
  mask_test = image_segementation_stuff[1]

  def conv_block(input, num_filters):
      x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(input)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation("relu")(x)

      x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation("relu")(x)
      return x

  def encoder_block(input, num_filters):
      x = conv_block(input, num_filters)
      p = tf.keras.layers.MaxPool2D((2, 2))(x)
      return x, p

  def decoder_block(input, skip_features, num_filters):
      x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
      x = tf.keras.layers.Concatenate()([x, skip_features])
      x = conv_block(x, num_filters)
      return x

  # main function
  def build_unet(input_shape):
      inputs = tf.keras.Input(input_shape)

      s1, p1 = encoder_block(inputs, 32)
      s2, p2 = encoder_block(p1, 64)
      s3, p3 = encoder_block(p2, 128)
      s4, p4 = encoder_block(p3, 256)
      s5, p5 = encoder_block(p4, 512)

      b1 = conv_block(p5, 512)

      d0 = decoder_block(b1, s5, 512)
      d1 = decoder_block(d0, s4, 256)
      d2 = decoder_block(d1, s3, 128)
      d3 = decoder_block(d2, s2, 64)
      d4 = decoder_block(d3, s1, 32)

      outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

      model = tf.keras.Model(inputs, outputs, name="U-Net")
      return model

  input_shape = (128, 128, 1) # (128, 128, 1)
  model = build_unet(input_shape)

  print(tf.keras.utils.plot_model(model, show_shapes=True))

  print(model.summary())

  # dice
  def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

  # iou
  def iou_coef(y_true, y_pred, smooth=1):
      intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3]) # changed [1,2,3] to [1] 
      union = K.sum(y_true,[1, 2, 3])+K.sum(y_pred,[1, 2, 3])-intersection
      iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
      return iou

  # customized optimizer
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=0.00009,
      decay_steps=10000,
      decay_rate=0.6)

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

  model.compile(
      optimizer=optimizer,
      loss=dice_loss,
      metrics=[iou_coef,'accuracy']
  )

  # train
  with tf.device('/device:GPU:0'):
      history = model.fit(
          x=pic_train, # X: the data the model use to learn
          y=mask_test.reshape((len(mask_test), 128, 128, 1)), # Y: the target the model try to predict
          validation_split=0.2, # a ratio of percentage that the model uses for validating
          batch_size=70,
          epochs=200
      )
  
  return {
    'model': model,
    'history': history
    
  }
