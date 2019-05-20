import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
import pandas as pd

ALP2Num = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9,
        "K":10, "L":11, "M":12, "N":13, "O":14, "P":15, "Q":16, "R":17, "S":18, "T":19,
        "U":20, "V":21, "W":22, "X":23, "Y":24, "Z":25, "del":26, "space":27, 'nothing':28
        }

def load_data_lbl(fname=[],imgW=150,imgH=150,totalImgNum=84000,sampleStep=3):
    train_flds = glob.glob(fname)
    dataX = np.zeros((int(totalImgNum/sampleStep),imgH,imgW,3)).astype('float32')
    dataLbl = []
    imgIdx = 0
    fileIdx = 0
    for fld in train_flds:
        imgfiles = glob.glob(fld + '/*.jpg')
        if len(imgfiles) == 0: continue
        rootName, fldName = fld.split('\\')
        # skip "nothing"
        if fldName == 'nothing': continue
        print("current fld name: ", fldName)
        for fidx, imgfile in enumerate(imgfiles):
            if fileIdx % sampleStep == 0:
                dataX[imgIdx] = img_to_array(load_img(imgfile, target_size=IMG_DIM))
                dataLbl.append(fldName)
                imgIdx += 1
            fileIdx += 1

            # if fidx % 10 == 0: print('{:03d}'.format(2
            # fidx) + "/ " + '{:05d}'.format(len(imgfiles)))

    datay = np.zeros(len(dataLbl)).astype('int')
    for i, lbl in enumerate(dataLbl):
        datay[i] = ALP2Num[lbl]

    print("len(dataX): ", len(dataX))
    print("imgIdx: ", imgIdx)

    return dataX, datay

def load_data_lbl_old2(fname=[],imgW=150,imgH=150):
    train_flds = glob.glob(fname)
    dataX = []
    dataLbl = []
    initalFlg = 1
    for fld in train_flds:
        imgfiles = glob.glob(fld + '/*.jpg')
        if len(imgfiles) == 0: continue
        rootName, fldName = fld.split('\\')
        # skip "nothing"
        if fldName == 'nothing': continue
        print("")
        print("current fld name: ", fldName)
        for fidx, imgfile in enumerate(imgfiles):
            if initalFlg:
                dataX = np.zeros((1, imgH, imgW, 3)).astype('float32')
                initalFlg = 0
                continue
            dataX = np.vstack((dataX,img_to_array(load_img(imgfile, target_size=IMG_DIM)).reshape(1, imgH, imgW, 3)))
            dataLbl.append(fldName)

            # if fidx in [len(imgfiles)/4, len(imgfiles)*2/4, len(imgfiles)*3/4]:
            #     print(".", end=' ')
            if fidx % 10 == 0: print('{:03d}'.format(fidx) + "/ " + '{:05d}'.format(len(imgfiles)))

    datay = np.zeros(len(dataLbl)).astype('int')
    for i, lbl in enumerate(dataLbl):
        datay[i] = ALP2Num[lbl]

    # dataX = np.array(dataX)
    # datay = np.array(datay).astype('int')
    #
    # # delete "nothing" img
    # idxsNotDelete = (datay != 28)
    # datay = datay[idxsNotDelete]
    # dataX = dataX[idxsNotDelete]

    return dataX, datay

def load_data_lbl_old(fname=[],imgW=150,imgH=150):
    train_flds = glob.glob(fname)
    dataX = []
    dataLbl = []
    for fld in train_flds:
        imgfiles = glob.glob(fld + '/*.jpg')
        if len(imgfiles) == 0: continue
        rootName, fldName = fld.split('\\')
        for imgfile in imgfiles:
            dataX.append(img_to_array(load_img(imgfile, target_size=IMG_DIM)))
            dataLbl.append(fldName)

    datay = np.zeros(len(dataLbl))
    for i, lbl in enumerate(dataLbl):
        datay[i] = ALP2Num[lbl]

    dataX = np.array(dataX)
    datay = np.array(datay).astype('int')

    # delete "nothing" img
    idxsNotDelete = (datay != 28)
    datay = datay[idxsNotDelete]
    dataX = dataX[idxsNotDelete]

    return dataX, datay

imgW = 150
imgH = 150
IMG_DIM = (imgW, imgH)

X_train, Y_train = load_data_lbl('asl_alphabet_train/*',imgW=150,imgH=150,totalImgNum=84000,sampleStep=3)
X_test, Y_test = load_data_lbl('asl-alphabet-test/*',imgW=150,imgH=150,totalImgNum=840,sampleStep=1)

# one-hot encoding
enc = OneHotEncoder(sparse=False, dtype=np.float32)
Ytrain_ohe = enc.fit_transform(Y_train.reshape(-1,1))
Ytest_ohe = enc.fit_transform(Y_test.reshape(-1,1))
total_classes = Ytrain_ohe.shape[1]

# img Augmentation
batchSize = 30
trainSplSize = len(X_train)
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                   fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(X_train, Ytrain_ohe, batch_size=batchSize)
val_generator = val_datagen.flow(X_test, Ytest_ohe, batch_size=20)

del X_train, X_test, Y_train, Y_test, Ytrain_ohe, Ytest_ohe

# build CNN
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, BatchNormalization,GlobalAveragePooling2D
from keras.models import Sequential
from keras import optimizers
# load VGG16
from keras.applications import vgg16
from keras.models import Model
import keras

vgg = vgg16.VGG16(include_top=False, weights='imagenet',input_shape=(imgW,imgH,3))

x = vgg.output
# x = GlobalAveragePooling2D()(x)
x = keras.layers.Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
predictions = Dense(total_classes, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=predictions)

# set trainable
for layer in model.layers[:15]:
   layer.trainable = False
for layer in model.layers[15:]:
   layer.trainable = True

layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
print("layer, layer.name, layer.trainable")
for layer in layers:
    print(layer)

Epochs = 10
stepPerEpochs = int(trainSplSize/batchSize)
model.compile(Adam(lr=.003), loss='categorical_crossentropy', metrics=['accuracy'])
print("model.summary()")
print(model.summary())
history = model.fit_generator(train_generator, steps_per_epoch=stepPerEpochs, epochs=Epochs,
                              validation_data=val_generator, validation_steps=50,
                              verbose=1)

model.save('ASL_vgg16ft_r4.h5')

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
plt.savefig('myKerasFinetune_r4.png')