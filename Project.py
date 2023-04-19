import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from PIL import Image

contacts = ['1232', '2354', '4365', '5451', '3123', '6547']

x = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]

print(pd.Series(y).value_counts())

classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
no_ofclasses = len(classes)

X_train, X_test, Y_train, Y_test = train_test_split(x, y , random_state=9,train_size=3500, test_size=500)

def get_Prediction(image):
    im_pil = Image.open(x)
    img_bw = im_pil.convert('L')
    img_bw_re = img_bw.resize((22,30), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(img_bw_re, pixel_filter)
    img_bw_re_inv_sca = np.clip(img_bw_re-min_pixel, 0, 255)
    max_pixel = np.max(img_bw_re)


