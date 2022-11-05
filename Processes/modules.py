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
from sklearn.model_selection import train_test_split

def acquire_kaggle_data(
    covid_imgs = covid_imgs,
    covid_masks = covid_masks,
    normal_imgs = normal_imgs_i,
    normal_masks = normal_masks_i,
    pneumonia_imgs = pneumonia_imgs_i,
    pneumonia_masks = pneumonia_masks_i,
    data_folder = data_folder_i
):
    ## Get kaggle data

    # import covid images
    os.chdir(covid_imgs)
    sorted_covid_data_ = os.listdir()
    # sort images
    sorted_covid_data_ = sorted(sorted_covid_data_, key=lambda x: int(x[6:-4]))

    covid_image_data_ = [] # list
    for i in tqdm(sorted_covid_data_[0:3616 ]):
        img = iio.imread(i)
        img = cv.resize(img, (128, 128)) # resize as (128, 128)
        img = np.array(img).tolist()
        covid_image_data_.append(img)
    covid_image_data_ = np.asarray(covid_image_data_)

    # import covid masks
    os.chdir(covid_masks)
    sorted_covid_masks_ = os.listdir()
    # sort masks
    sorted_covid_masks_ = sorted(sorted_covid_masks_, key=lambda x: int(x[6:-4]))

    covid_mask_data_ = [] # list
    for i in tqdm(sorted_covid_masks_[0:3616 ]):
        img = iio.imread(i)
        img = cv.resize(img, (128, 128))
        img = np.array(img)
        covid_mask_data_.append((img[:,:,0]).tolist())
    covid_mask_data_ = np.asarray(covid_mask_data_)

    # save covid data in pickle file

    os.chdir(data_folder)

    a = (covid_image_data_, covid_mask_data_)
    with open('covid_image_and_mask_.pkl', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # import normal images
    os.chdir(normal_imgs)
    sorted_normal_data_ = os.listdir()
    # sort normal images
    sorted_normal_data_ = sorted(sorted_normal_data_, key=lambda x: int(x[7:-4]))

    normal_image_data_ = [] # list
    for i in tqdm(sorted_normal_data_[0:3616 ]):
        img = iio.imread(i)
        img = cv.resize(img, (128, 128))
        img = np.array(img).tolist()
        normal_image_data_.append(img)
    normal_image_data_ = np.asarray(normal_image_data_)

    # import normal masks
    os.chdir(normal_masks)
    sorted_normal_masks_ = os.listdir()
    # sort normal masks
    sorted_normal_masks_ = sorted(sorted_normal_masks_, key=lambda x: int(x[7:-4]))

    normal_mask_data_ = [] # list
    for i in tqdm(sorted_normal_masks_[0:3616 ]):
        img = iio.imread(i)
        img = cv.resize(img, (128, 128))
        img = np.array(img)
        normal_mask_data_.append((img[:,:,0]).tolist())
    normal_mask_data_ = np.asarray(normal_mask_data_)

    # save normal data in pickle file

    os.chdir(data_folder)

    b = (normal_image_data_, normal_mask_data_)
    with open('normal_image_and_mask_.pkl', 'wb') as handle:
        pickle.dump(b, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # import pneumonia images
    os.chdir(pneumonia_imgs)
    sorted_pneumonia_data_ = os.listdir()
    # sort images
    sorted_pneumonia_data_ = sorted(sorted_pneumonia_data_, key=lambda x: int(x[16:-4]))

    pneumonia_image_data_ = [] # list
    for i in tqdm(sorted_pneumonia_data_):
        img = iio.imread(i)
        img = cv.resize(img, (128, 128))
        if len(img.shape) > 2:
            img = img[:,:,0]
            img = np.asarray(img).tolist()
        else:
            img = np.asarray(img).tolist()
        pneumonia_image_data_.append(img)
    pneumonia_image_data_ = np.asarray(pneumonia_image_data_)

    # import pneumonia masks
    os.chdir(pneumonia_masks)
    sorted_pneumonia_masks_ = os.listdir()
    # sort pneumonia masks
    sorted_pneumonia_masks_ = sorted(sorted_pneumonia_masks_, key=lambda x: int(x[16:-4]))

    pneumonia_mask_data_ = [] # list
    for i in tqdm(sorted_pneumonia_masks_):
        img = iio.imread(i)
        img = cv.resize(img, (128, 128))
        img = np.array(img)
        pneumonia_mask_data_.append((img[:,:,0]).tolist())
    pneumonia_mask_data_ = np.asarray(pneumonia_mask_data_)

    os.chdir(data_folder)

    c = (pneumonia_image_data_, pneumonia_mask_data_)
    with open('pneumonia_image_and_mask_.pkl', 'wb') as handle:
        pickle.dump(c, handle, protocol=pickle.HIGHEST_PROTOCOL)

def segmentation_data(
    covid_imgs = covid_imgs_i,
    covid_masks = covid_masks_i,
    normal_imgs = normal_imgs_i,
    normal_masks = normal_masks_i,
    pneumonia_imgs = pneumonia_imgs_i,
    pneumonia_masks = pneumonia_masks_i,
    data_folder = data_folder_i
):
    ## Get kaggle data

    # import covid images
    os.chdir(covid_imgs)
    sorted_covid_data_ = os.listdir()
    # sort images
    sorted_covid_data_ = sorted(sorted_covid_data_, key=lambda x: int(x[6:-4]))

    covid_image_data_ = [] # list
    for i in tqdm(sorted_covid_data_[0:1345]):
        img = iio.imread(i)
        img = cv.resize(img, (128, 128)) # resize as (128, 128)
        img = np.array(img).tolist()
        covid_image_data_.append(img)
    covid_image_data_ = np.asarray(covid_image_data_)

    # import covid masks
    os.chdir(covid_masks)
    sorted_covid_masks_ = os.listdir()
    # sort masks
    sorted_covid_masks_ = sorted(sorted_covid_masks_, key=lambda x: int(x[6:-4]))

    covid_mask_data_ = [] # list
    for i in tqdm(sorted_covid_masks_[0:1345]):
        img = iio.imread(i)
        img = cv.resize(img, (128, 128))
        img = np.array(img)
        covid_mask_data_.append((img[:,:,0]).tolist())
    covid_mask_data_ = np.asarray(covid_mask_data_)

    # save covid data in pickle file

    os.chdir(data_folder)

    a = (covid_image_data_, covid_mask_data_)
    with open('covid_image_and_mask_final_2.pkl', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # import normal images
    os.chdir(normal_imgs)
    sorted_normal_data_ = os.listdir()
    # sort normal images
    sorted_normal_data_ = sorted(sorted_normal_data_, key=lambda x: int(x[7:-4]))

    normal_image_data_ = [] # list
    for i in tqdm(sorted_normal_data_[0:1345]):
        img = iio.imread(i)
        img = cv.resize(img, (128, 128))
        img = np.array(img).tolist()
        normal_image_data_.append(img)
    normal_image_data_ = np.asarray(normal_image_data_)

    # import normal masks
    os.chdir(normal_masks)
    sorted_normal_masks_ = os.listdir()
    # sort normal masks
    sorted_normal_masks_ = sorted(sorted_normal_masks_, key=lambda x: int(x[7:-4]))

    normal_mask_data_ = [] # list
    for i in tqdm(sorted_normal_masks_[0:1345]):
        img = iio.imread(i)
        img = cv.resize(img, (128, 128))
        img = np.array(img)
        normal_mask_data_.append((img[:,:,0]).tolist())
    normal_mask_data_ = np.asarray(normal_mask_data_)

    # save normal data in pickle file

    os.chdir(data_folder)

    b = (normal_image_data_, normal_mask_data_)
    with open('normal_image_and_mask_.pkl', 'wb') as handle:
        pickle.dump(b, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def image_cropping(
    data_folder = data_folder_i
):
    os.chdir(data_folder)

    with open('covid_image_and_mask_final_2.pkl', 'rb') as handle:
        covid_images_and_masks = pickle.load(handle)

    with open('normal_image_and_mask_final_2.pkl', 'rb') as handle:
        normal_images_and_masks = pickle.load(handle)

    with open('pneumonia_image_and_mask_.pkl', 'rb') as handle:
        pneumonia_images_and_masks = pickle.load(handle)

    cropped_covid_images_ = []
    for i in tqdm(range(len(covid_images_and_masks[0]))):
        cropped_covid_images_.append(covid_images_and_masks[0][i]*covid_images_and_masks[1][i]/255)
        cropped_covid_images_ = np.asarray(cropped_covid_images_)

    os.chdir(data_folder)

    d = cropped_covid_images_
    with open('cropped_covid_images_final_2.pkl', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    cropped_normal_images_ = []
    for i in tqdm(range(len(normal_images_and_masks[0]))):
        cropped_normal_images_.append(normal_images_and_masks[0][i]*normal_images_and_masks[1][i]/255)
        cropped_normal_images_ = np.asarray(cropped_normal_images_)

    os.chdir(data_folder)

    e = cropped_normal_images_
    with open('cropped_normal_images_final_2.pkl', 'wb') as handle:
        pickle.dump(e, handle, protocol=pickle.HIGHEST_PROTOCOL)

    cropped_pneumonia_images_ = []
    for i in tqdm(range(len(pneumonia_images_and_masks[0]))):
        cropped_pneumonia_images_.append(pneumonia_images_and_masks[0][i]*pneumonia_images_and_masks[1][i]/255)
        cropped_pneumonia_images_ = np.asarray(cropped_pneumonia_images_)

    os.chdir(data_folder)

    f = cropped_pneumonia_images_
    with open('cropped_pneumonia_images.pkl', 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)

def shuffling_and_setting_data(
    data_folder = data_folder_i
):
    os.chdir(data_folder)

    with open('cropped_covid_images_final_2.pkl', 'rb') as handle:
        cropped_covid_stuff_ = pickle.load(handle)

    with open('cropped_normal_images_final_2.pkl', 'rb') as handle:
        cropped_normal_stuff_ = pickle.load(handle)

    with open('cropped_pneumonia_images.pkl', 'rb') as handle:
        cropped_pneumonia_stuff_ = pickle.load(handle)

    all_IMAGES = [] 
    all_LABELS = []
    for i in tqdm(range(len(cropped_covid_stuff_))):
        all_IMAGES.append(cropped_covid_stuff_[i])
        all_LABELS.append(0)
    for i in tqdm(range(len(cropped_normal_stuff_))):
        all_IMAGES.append(cropped_normal_stuff_[i])
        all_LABELS.append(1)
    for i in tqdm(range(len(cropped_pneumonia_stuff_))):
        all_IMAGES.append(cropped_pneumonia_stuff_[i])
        all_LABELS.append(2)

    pic_train, pic_test, label_train, label_test = train_test_split(np.asarray(all_IMAGES), np.asarray(all_LABELS), test_size= 0.2, train_size = 0.8, random_state=42)

    with open('test_everything_final_2.pkl', 'wb') as handle:
        pickle.dump((pic_test, tf.keras.utils.to_categorical(label_test)), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('train_everything_final_2.pkl', 'wb') as handle:
        pickle.dump((pic_train, tf.keras.utils.to_categorical(label_train)), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def segmentation_data_2(
    data_folder = data_folder_i
):
    with open('covid_image_and_mask_.pkl', 'rb') as handle:
        covid_everything = pickle.load(handle)

    with open('normal_image_and_mask_.pkl', 'rb') as handle:
        normal_everything = pickle.load(handle)

    with open('pneumonia_image_and_mask_.pkl', 'rb') as handle:
        pneumonia_everything = pickle.load(handle)

    covid_images, covid_masks = covid_everything
    covid_images = covid_images.tolist()
    covid_masks = covid_masks.tolist()

    normal_images, normal_masks = normal_everything
    normal_images = normal_images.tolist()
    normal_masks = normal_masks.tolist()

    pneumonia_images, pneumonia_masks = pneumonia_everything
    pneumonia_images = pneumonia_images.tolist()
    pneumonia_masks = pneumonia_masks.tolist()

    big_boi_images = covid_images+normal_images+pneumonia_images
    big_boi_masks = covid_masks+normal_masks+pneumonia_masks

    big_boi_images = np.asarray(big_boi_images)
    big_boi_masks = np.asarray(big_boi_masks)

    plt.imshow(big_boi_images[2020], cmap = 'binary')

    plt.imshow(big_boi_masks[2020], cmap = 'binary')

    pic_train = big_boi_images/255.

    mask_test = big_boi_masks/255.

    with open('image_segementation_data_final_og.pkl', 'wb') as handle:
        pickle.dump((pic_train, mask_test), handle, protocol=pickle.HIGHEST_PROTOCOL)
