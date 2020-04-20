# all necessary imports.
import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import glob
import json
from os import listdir
from PIL import Image

import argparse
import tensorflow_hub as hub
# print('Using:')
# print('\t\u2022 TensorFlow version:', tf.__version__)
# print('\t\u2022 tf.keras version:', tf.keras.__version__)

# Initiate variables with default values
modelpath = './my_model.h5'
filepath = './label_map.json'    
arch=''
image_path = './test_images/wild_pansy.jpg'
topk = 5
IMAGE_RES = 224


# Set up parameters for entry in command line
parser = argparse.ArgumentParser()
parser.add_argument('-m','--modelpath', action='store',type=str, help='Name of trained model to be loaded and used for predictions.')
parser.add_argument('-i','--image_path',action='store',type=str, help='Location of image to predict e.g. test/image.jpg')
parser.add_argument('-k', '--topk', action='store',type=int, help='Select number of classes you wish to see in descending order.')
parser.add_argument('-j', '--json', action='store',type=str, help='Define name of json file holding class names.')

args = parser.parse_args()

# Select parameters entered in command line
if args.modelpath:
    modelpath = args.modelpath
if args.image_path:
    image_path = args.image_path
if args.topk:
    topk = args.topk
if args.json:
    filepath = args.json

    
with open(filepath, 'r') as f:
    class_names = json.load(f)

def process_image(image_path):
    image = np.squeeze(image_path)
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image

def predict(image_path, model, topk=5):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    prediction = model.predict(np.expand_dims(processed_test_image, axis=0))
    top_values, top_indices = tf.math.top_k(prediction, topk)
    # print("These are the top propabilities",top_values.numpy()[0])
    top_classes = [class_names[str(value+1)] for value in top_indices.cpu().numpy()[0]]
    # print('Of these top classes', top_classes)
    
    return top_values.numpy()[0], top_classes

model=tf.keras.models.load_model(modelpath,custom_objects={'KerasLayer':hub.KerasLayer})
# print(model.summary())

print('-' * 50)
probs, classes = predict(image_path, model, topk)

print("Image predected is : {}".format(classes[0]))
print('-' * 50)

print("These are the top propabilities")
for i in range(topk):
    print(" {} : {}".format(classes[i],probs[i]))

