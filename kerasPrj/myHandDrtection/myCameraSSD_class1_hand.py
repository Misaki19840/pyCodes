import cv2
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from matplotlib import pyplot as plt

import sys
sys.path.append('../ssd_keras')
from keras_loss_function.keras_ssd_loss import SSDLoss
from models_ssd.keras_ssd300 import ssd_300

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
classes = ['background',
           'hand']

# Set a few configuration parameters.
img_height = 300
img_width = 300
n_classes = 1
model_mode = 'inference'

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=n_classes,
                mode=model_mode,
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# load weight
weights_path = 'mypartsDetectionSSD_class1_epoch-01_loss-4.9658_val_loss-6.6276.h5'
model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
print("model summary: ", model.summary())

# prepare capture
cam = cv2.VideoCapture(0)
while True:
    _, img = cam.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    camImgH = img_rgb.shape[0]
    camImgW = img_rgb.shape[1]
    img_rgb_resized = cv2.resize(img_rgb, (img_width, img_height))
    testImg = img_rgb_resized.reshape(-1, img_height, img_width, 3).astype('float32')
    y_pred = model.predict(testImg)

    i = 0
    confidence_threshold = 0.5
    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
    print("Predicted boxes:\n")
    print('    class    conf  xmin    ymin    xmax    ymax')
    print(y_pred_thresh[0])

    for box in y_pred_thresh[i]:
        class_id = box[0]
        confidence = box[1]
        xmin = box[2] * camImgW / img_width
        ymin = box[3] * camImgH / img_height
        xmax = box[4] * camImgW / img_width
        ymax = box[5] * camImgH / img_height
        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)
        color = colors[int(box[0])]
        tlabel = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        cv2.putText(img, tlabel, (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.imshow('PUSH ENTER KEY', img)
    if cv2.waitKey(1) == 13: break

cam.release()
cv2.destroyAllWindows()