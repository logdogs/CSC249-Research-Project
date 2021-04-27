# coding: utf-8
"""
    Test file or on a subset of 2013 CASIA competition data

"""

import os
import csv
import codecs
import random
import copy
import argparse
import numpy as np
import tensorflow as tf
import PIL.Image as Image
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import cv2

# from scipy.misc import imread
from imageio import imread
from keras.utils.np_utils import to_categorical
from keras import backend
from the_model import model_8

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pred", action="store_true")
parser.add_argument("infile", nargs="?", type=argparse.FileType('rb'))
args = parser.parse_args()

random.seed(888)
np.random.seed(888)
#tf.set_random_seed(888)
tf.random.set_seed(888)

IMG_SIZE = 96

Pred_Details = False
if args.pred or args.infile:
    Pred_Details = True

TEST_PATH = os.path.join("data", "test")
WEIGHTS_PATH = os.path.join("data", "weights08.h5")
LABELS_PATH = os.path.join("data", "labels.txt")

label_file = codecs.open(LABELS_PATH, "r", "UTF-8")
klasses = [a.strip() for a in label_file.readlines()]
label_file.close()

model = model_8(IMG_SIZE, len(klasses))
model.load_weights(WEIGHTS_PATH)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if args.infile:
    test_data = np.ndarray([1,IMG_SIZE,IMG_SIZE],dtype=np.uint8)
    test_data[0] = imread(args.infile) 

else:
    label_pngs = []
    for k, v in enumerate(klasses):
        for png in os.listdir(os.path.join(TEST_PATH, v)):
            label_pngs.append((k, v, png))

    print("Total number of test samples:", len(label_pngs))
    test_data = np.ndarray([len(label_pngs), IMG_SIZE, IMG_SIZE],
                           dtype=np.uint8)
    test_label = np.ndarray([len(label_pngs)], dtype=np.uint32)

    i = 0
    for label_png in label_pngs:
        fimg = open(os.path.join(TEST_PATH, label_png[1], label_png[2]), 'rb')
        test_data[i] = imread(fimg)
        test_label[i] = label_png[0]
        fimg.close()
        i += 1

    y_test = to_categorical(test_label)

x_test = test_data.reshape(test_data.shape[0],
                           test_data.shape[1],
                           test_data.shape[2],
                           1)
x_test = x_test.astype(np.float32)
x_test /= 255.0


def top_predictions(n, pred):
    tops = []
    pred_copy = copy.copy(pred)
    for j in range(n):
        i = np.argmax(pred_copy)
        tops.append((klasses[i], "%.2f" % (pred_copy[i] * 100)))
        pred_copy[i] = 0

    return tops

top = ''
if args.infile:
    preds = model.predict(x_test)
    top3 = top_predictions(3,preds[0])
    top = top3[0]
    print("Prediction: top3:", top_predictions(3, preds[0]))

elif Pred_Details:
    preds = model.predict(x_test)
    for k, v in enumerate(preds):
        p = np.argmax(v)
        if p != test_label[k]:
            top3 = top_predictions(3,v)
            top = top3[0]
            print("Wrong prediction: top3:", top3,
                  "label/file:", label_pngs[k][1], label_pngs[k][2])
    
else:
    loss, acc = model.evaluate(x_test, y_test, batch_size=64)
    print("Loss:", loss)
    print("Accuracy:", acc)


def get_structure(character):
    #file = open("cnn_output_character.txt", 'r', encoding='utf8')
    #character = file.read()
    with open('cld2.csv', newline='', encoding='utf8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if row['C1'] == character:
                return row['C1Structure'], row['C1SR'], row['C1PR']


def bound_image(bin_image):
    max_x = 0
    max_y = 0
    min_x = 10000000
    min_y = 10000000
    for i in range(bin_image.shape[0]):
        for j in range(bin_image.shape[1]):
            if bin_image[i,j] > 0:
                if i > max_x:
                    max_x = i
                if i < min_x:
                    min_x = i
                if j > max_y:
                    max_y = j
                if j < min_y:
                    min_y = j
    image_prime = bin_image[min_x:max_x, min_y:max_y]
    return image_prime

def grayscale(image):
    return np.dot(image[:,:], [1/3,1/3,1/3])


top_character = list(top)[0]
file = open('cnn_output_character.txt', 'w', encoding='utf8')
file.write(top_character)
## Make canvas and set the color
img = np.zeros((200,400,3),np.uint8)
b,g,r,a = 255,255,255,0

## Use simsum.ttc to write Chinese.
fontpath = "./simsun.ttc" # <== 这里是宋体路径
font = ImageFont.truetype(fontpath, 175)
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
#draw.text((50, 80),  "端午节就要到了", font = font, fill = (b, g, r, a))
draw.text((20, 20), top_character, font = font, fill = (b, g, r, a))
img = np.array(img_pil)

cropped = bound_image(grayscale(img))
resized = cv2.resize(cropped,(96,96),interpolation=cv2.INTER_CUBIC)

## Display 
# cv2.imshow("res", resized)
# cv2.waitKey()
# cv2.destroyAllWindows()
cv2.imwrite("res.png", resized)


backend.clear_session()
