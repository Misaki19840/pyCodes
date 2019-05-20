import cv2
import numpy as np
from keras.models import load_model

ALP2Num = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9,
        "K":10, "L":11, "M":12, "N":13, "O":14, "P":15, "Q":16, "R":17, "S":18, "T":19,
        "U":20, "V":21, "W":22, "X":23, "Y":24, "Z":25, "del":26, "space":27
        }
Num2ALF = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z", "del", "space"
        ]

model = load_model('ASL_vgg16ft_r5.h5')

# prepare capture
cam = cv2.VideoCapture(0)
while True:
    # 画像を取得 --- (*2)
    _, img = cam.read()
    imgflip = cv2.flip(img,1)

    roi = (250, 200, 400, 350)

    #rect_img = imgflip.copy()
    rect_img = img.copy()
    cv2.rectangle(rect_img, (roi[2],roi[3]), (roi[0],roi[1]), (255, 0, 0), 2)

    #
    s_roi = img[roi[1]: roi[3], roi[0]: roi[2]]

    testImg = s_roi.reshape(-1, s_roi.shape[0], s_roi.shape[1], s_roi.shape[2]).astype('float32')
    testImg /= 255
    predictions = model.predict(testImg)
    print(predictions)
    maxArg = np.argmax(predictions[0])
    print("maxArg")
    print(maxArg)
    print("result")
    print(Num2ALF[maxArg])

    # 　出力画像の同じ箇所に埋め込み
    rect_img[roi[1]: roi[3], roi[0]: roi[2]] = s_roi

    if predictions[0][maxArg] > 0.5:
        text = Num2ALF[maxArg]
        cv2.putText(rect_img,text,(roi[1]-10,roi[2]-10),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0))

    cv2.imshow('PUSH ENTER KEY', rect_img)
    if cv2.waitKey(1) == 13: break
cam.release()
cv2.destroyAllWindows()