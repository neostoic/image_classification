#!/usr/bin/python
from PIL import Image
from PIL import ImageOps
import os, sys

path = r"D:\Yelp\photos"
labels = [r'\drink\\', r'\food\\', r'\inside\\', r'\outside\\', r'\menu\\']

def resize():
    for label in labels:
        dirs = os.listdir(path + label)
        for item in dirs:
            if os.path.isfile(path+label+item):
                im = Image.open(path+label+item)
                f, e = os.path.splitext(item)
                im = ImageOps.fit(im, (224, 224), Image.ANTIALIAS)
                im.save(path + r'\resized' + label + f + '_224by224.jpg', 'JPEG')

resize()