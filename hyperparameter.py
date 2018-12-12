# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:51:27 2018

@author: wangh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:39:30 2018

@author: shaohanwang
"""
from numpy.random import seed
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
from numpy.random import seed
np.random.seed(7)
train_data=pd.read_csv('after.csv')
train_data=train_data.dropna()
x=train_data.iloc[:,1:-1]
y= train_data.iloc[:,-1]

from keras.layers import Dropout
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
input_dim = np.size(x_train, 1)
sc2=StandardScaler()

#pca = PCA(n_components=10)
#x_train=pca.fit_transform(x_train)
#x_test=pca.transform(x_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
# def build_classifier():
#    classifier = Sequential()
#    classifier.add(Dense(output_dim=42,init='uniform',activation='relu',input_dim=input_dim))
#
#    classifier.add(Dense(output_dim=42,init='uniform',activation='relu'))
#    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
#    classifier.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
#    return classifier
# classifier= KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)

from sklearn.metrics import confusion_matrix

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim=52,init='uniform',activation='relu',input_dim=input_dim))

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=10,epochs=50)
y_predict=classifier.predict(x_test)
y_probability2=y_predict
y_predict=(y_predict>0.5)
y_test2=y_test


from sklearn.metrics import confusion_matrix, classification_report
cm2=confusion_matrix(y_test,y_predict)
np.savetxt("confusion_matrix(hyperparameter).csv", cm2, delimiter=",")
import pathlib

report2=classification_report(y_test,y_predict,digits=5)
pathlib.Path("report(hyperparameter).txt").write_text(f"hyper:{report2}")

print(classification_report(y_test,y_predict,digits=5))

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


from sklearn.metrics import roc_curve, auc
lw=2
auc2 = roc_auc_score(y_test, y_probability2)
# calculate roc curve
fpr2, tpr2, thresholds = roc_curve(y_test, y_probability2)
plt.figure()
plt.plot(fpr2, tpr2, color='darkorange', lw=1, label='ROC curve (area = %0.5f)' % auc2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic(hyperparameter)')
plt.legend(loc="lower right")
plt.savefig('hyper_roc.jpg')

plt.show()
yvalue=y_test.values
from pandas.tools.plotting import table

df=pd.DataFrame(data=cm2, columns = ["predicted missed playoffs", "predicted making playoffs"])
df.rename(index={0:'actual missed playoffs',1:'actual making playoffs'}, inplace=True)

fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
tabla = table(ax, df, loc='upper right', colWidths=[0.17]*len(df.columns))  # where df is your data frame
tabla.auto_set_font_size(False) # Activate set fontsize manually
tabla.set_fontsize(10) # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1.2) # change size table
plt.savefig('confusion(hypersearch).png', transparent=True)
import PIL
import PIL.Image
import PIL.ImageFont
import PIL.ImageOps
import PIL.ImageDraw

PIXEL_ON = 0  # PIL color to use for "on"
PIXEL_OFF = 255  # PIL color to use for "off"


def main():
    image = text_image('report(hyperparameter).txt')
    image.show()
    image.save('classification_report(hyper).png')


def text_image(text_path, font_path=None):
    """Convert text file to a grayscale image with black characters on a white background.

    arguments:
    text_path - the content of this file will be converted to an image
    font_path - path to a font file (for example impact.ttf)
    """
    grayscale = 'L'
    # parse the file into lines
    with open(text_path) as text_file:  # can throw FileNotFoundError
        lines = tuple(l.rstrip() for l in text_file.readlines())

    # choose a font (you can see more detail in my library on github)
    large_font = 20  # get better resolution with larger size
    font_path = font_path or 'cour.ttf'  # Courier New. works in windows. linux may need more explicit path
    try:
        font = PIL.ImageFont.truetype(font_path, size=large_font)
    except IOError:
        font = PIL.ImageFont.load_default()
        print('Could not use chosen font. Using default.')

    # make the background image based on the combination of font and lines
    pt2px = lambda pt: int(round(pt * 96.0 / 72))  # convert points to pixels
    max_width_line = max(lines, key=lambda s: font.getsize(s)[0])
    # max height is adjusted down because it's too large visually for spacing
    test_string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    max_height = pt2px(font.getsize(test_string)[1])
    max_width = pt2px(font.getsize(max_width_line)[0])
    height = max_height * len(lines)  # perfect or a little oversized
    width = int(round(max_width + 40))  # a little oversized
    image = PIL.Image.new(grayscale, (width, height), color=PIXEL_OFF)
    draw = PIL.ImageDraw.Draw(image)

    # draw each line of text
    vertical_position = 5
    horizontal_position = 5
    line_spacing = int(round(max_height * 0.8))  # reduced spacing seems better
    for line in lines:
        draw.text((horizontal_position, vertical_position),
                  line, fill=PIXEL_ON, font=font)
        vertical_position += line_spacing
    # crop the text
    c_box = PIL.ImageOps.invert(image).getbbox()
    image = image.crop(c_box)
    return image


if __name__ == '__main__':
    main()

