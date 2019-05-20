import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
import pandas as pd

imgW = 150
imgH = 150
IMG_DIM = (imgW, imgH)
train_flds = glob.glob('asl-alphabet-test/*')
dataX = []
dataLbl = []
for fld in train_flds:
    imgfiles = glob.glob(fld + '/*.jpg')
    if len(imgfiles) == 0: continue
    rootName, fldName = fld.split('\\')
    for imgfile in imgfiles:
        dataX.append(img_to_array(load_img(imgfile, target_size=IMG_DIM)))
        dataLbl.append(fldName)


ALP2Num = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9,
        "K":10, "L":11, "M":12, "N":13, "O":14, "P":15, "Q":16, "R":17, "S":18, "T":19,
        "U":20, "V":21, "W":22, "X":23, "Y":24, "Z":25, "del":26, "space":27, 'nothing':28
        }

datay = np.zeros(len(dataLbl))
for i, lbl in enumerate(dataLbl):
    datay[i] = ALP2Num[lbl]


dataX = np.array(dataX)
datay = np.array(datay).astype('int')

# delete "nothing" img
idxsNotDelete = (datay != 28)
datay = datay[idxsNotDelete]
dataX = dataX[idxsNotDelete]

X_train, X_test, Y_train, Y_test = train_test_split(dataX, datay,train_size=0.8,random_state=2)

# normalization
Xtrain_scaled = X_train.astype('float32')
Xtest_scaled  = X_test.astype('float32')
Xtrain_scaled /= 255
Xtest_scaled /= 255

# one-hot encoding
# Ytrain_ohe = pd.get_dummies(Y_train.reset_index(drop=True)).as_matrix()
# Ytest_ohe = pd.get_dummies(Y_test.reset_index(drop=True)).as_matrix()
enc = OneHotEncoder(sparse=False, dtype=np.float32)
Ytrain_ohe = enc.fit_transform(Y_train.reshape(-1,1))
Ytest_ohe = enc.fit_transform(Y_test.reshape(-1,1))

# img Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                   fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(X_train, Ytrain_ohe, batch_size=30)
val_generator = val_datagen.flow(X_test, Ytest_ohe, batch_size=20)


# output = vgg.layers[-1].output
# output = keras.layers.Flatten()(output)
# vgg_model = Model(vgg.input, output)
#
# vgg_model.trainable = False
# for layer in vgg_model.layers:
#     layer.trainable = False

# build CNN
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, BatchNormalization,GlobalAveragePooling2D
from keras.models import Sequential
from keras import optimizers
# load VGG16
from keras.applications import vgg16
from keras.models import Model
import keras

vgg = vgg16.VGG16(include_top=False, weights='imagenet',input_shape=(imgW,imgH,3))

# model = Sequential()
# model.add(vgg_model)
# input_shape = vgg_model.output_shape[1]
# model.add(Dense(512, activation='relu', input_dim=input_shape))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(28, activation='softmax'))

x = vgg.output
# x = GlobalAveragePooling2D()(x)
x = keras.layers.Flatten()(x)
x = Dense(512, activation='relu')(x)
# x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
# x = BatchNormalization()(x)
x = Dropout(0.3)(x)
total_classes = Ytrain_ohe.shape[1]
predictions = Dense(total_classes, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=predictions)

# set trainable
model.trainable = True
set_trainable = False
for layer in model.layers:
    if layer.name in ['block5_conv1', 'block4_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
print("layer, layer.name, layer.trainable")
for layer in layers:
    print(layer)

Epochs = 30
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
print("model.summary()")
print(model.summary())
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=Epochs,
                              validation_data=val_generator, validation_steps=50,
                              verbose=1)

model.save('ASL_vgg16ft.h5')

# results
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,Epochs+1))
ax1.plot(epoch_list, history.history['acc'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_acc'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, Epochs+1, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, Epochs+1, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

# plt.show()
plt.savefig('myKerasFinetune.png')