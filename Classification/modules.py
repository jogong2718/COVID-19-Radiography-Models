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

from google.colab import drive
drive.mount('/content/drive')

def classification_model(
  data_folder
):
  os.chdir(data_folder)

  with open('test_everything_final_2.pkl', 'rb') as handle:
      test_everything = pickle.load(handle)

  with open('train_everything_final_2.pkl', 'rb') as handle:
      train_everything = pickle.load(handle)

  test_data = test_everything[0]
  train_data = train_everything[0]

  train_data = train_data/255.
  test_data = test_data/255.

  train_data.shape # 7719
  test_data.shape # 858

  test_labels = test_everything[1]
  train_labels = train_everything[1]

  train_data = train_data.reshape((3228, 128, 128, 1))
  train_data = np.concatenate([train_data, train_data, train_data], 3)

  test_data = test_data.reshape((807, 128, 128, 1))
  test_data = np.concatenate([test_data, test_data, test_data], 3)

  conv_base = tf.keras.applications.DenseNet201(
          include_top=False,
          weights='imagenet',
          pooling='max')

  from tensorflow.keras import layers
  from tensorflow.keras import regularizers

  # input layers
  input_layer = tf.keras.Input(shape=(128, 128, 3))
  input_layer_aux = tf.keras.layers.Input(shape=(128, 128, 3))

  # DenseNet201 process
  conv1_aux = conv_base(input_layer_aux)
  conv3_aux = tf.keras.layers.Flatten()(conv1_aux)

  # encode
  conv1 = tf.keras.layers.Conv2D(32, kernel_size=(2,2))(input_layer)
  conv1_2 = tf.keras.layers.BatchNormalization()(conv1)
  conv2 = tf.keras.layers.Conv2D(32, kernel_size=(2,2))(conv1_2)
  conv2_2 = tf.keras.layers.BatchNormalization()(conv2)
  conv3 = tf.keras.layers.Conv2D(32, kernel_size=(2,2))(conv2_2)
  conv3_2 = tf.keras.layers.BatchNormalization()(conv3)
  conv4 = tf.keras.layers.Conv2D(32, kernel_size=(2,2))(conv3_2)
  conv4_2 = tf.keras.layers.BatchNormalization()(conv4)
  conv5 = tf.keras.layers.Conv2D(32, kernel_size=(2,2))(conv4_2)
  conv5_2 = tf.keras.layers.BatchNormalization()(conv5)
  conv6 = tf.keras.layers.Conv2D(32, kernel_size=(2,2))(conv5_2)
           
  #latent layer
  conv_latent = tf.keras.layers.BatchNormalization()(conv6)

  # decode
  deconv1 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2,2))(conv_latent)
  deconv2 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2,2))(deconv1)
  deconv3 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2,2))(deconv2)
  deconv4 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2,2))(deconv3)
  deconv5 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2,2))(deconv4)
  deconv6 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2,2))(deconv5)

  # output
  deconv_final = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(deconv6)

  # auxiliary output
  conv = tf.keras.layers.Conv2D(32, kernel_size=(2,2))(conv_latent)
  conv = tf.keras.layers.Conv2D(32, kernel_size=(2,2))(conv)
  conv = tf.keras.layers.BatchNormalization()(conv)
  conv = tf.keras.layers.MaxPool2D()(conv)

  conv = tf.keras.layers.Conv2D(32, kernel_size=(2,2))(conv)
  conv = tf.keras.layers.Conv2D(32, kernel_size=(2,2))(conv)
  conv = tf.keras.layers.BatchNormalization()(conv)
  conv = tf.keras.layers.MaxPool2D()(conv)

  aux_output = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(conv)
  aux_output = tf.keras.layers.Flatten()(aux_output)

  # concatenate
  concat_layer = tf.keras.layers.Concatenate()([aux_output, conv3_aux])
  clf_dense_1 = tf.keras.layers.Dense(128, activation = "relu")(concat_layer)

  clf_dropout_1 = tf.keras.layers.Dropout(0.2)(clf_dense_1)
  clf_dense_2 = tf.keras.layers.Dense(64, activation = "relu")(clf_dropout_1)
  clf_dropout_2 = tf.keras.layers.Dropout(0.3)(clf_dense_2)
  clf_dense_3 = tf.keras.layers.Dense(32, activation = "relu")(clf_dropout_2)
  clf_dropout_3 = tf.keras.layers.Dropout(0.2)(clf_dense_3)
  aux_output_dense = tf.keras.layers.Dense(3, activation = "softmax")(clf_dropout_3)

  # final model
  final_model = tf.keras.models.Model(inputs = [input_layer, input_layer_aux], outputs = [deconv_final, aux_output_dense])

  print(final_model.summary())

  print(tf.keras.utils.plot_model(final_model, show_shapes=True))

  # Adam Optimizer
  opt = tf.keras.optimizers.Adam(0.0001)

  final_model.compile(
      optimizer = opt,
      loss = "categorical_crossentropy",
      metrics = ["accuracy", tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])]
  )

  with tf.device('/device:GPU:0'):
    history = final_model.fit(
        [train_data, train_data], [train_data[:, :, :, 0:1], train_labels],
        validation_data = ([test_data, test_data], [test_data[:, :, :, 0:1], test_labels]),
        batch_size = 41,
        epochs = 50
    )
           
  return {
        'model': final_model,
        'history': history
    }
           
