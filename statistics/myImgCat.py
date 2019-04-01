import cv2
import numpy as np

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)

im1Name = "imgfld/ookawa918IMGL1370_TP_V.jpg_beforeafterwarp.jpg"
im2Name = "imgfld/ookawa918IMGL1370_TP_V.jpg_output_p5.jpg"
im3Name = "imgfld/ookawa918IMGL1370_TP_V.jpg_beforewarp.jpg"
im4Name = "imgfld/ookawa918IMGL1370_TP_V.jpg_afterwarp.jpg"
im5Name = "imgfld/ookawa918IMGL1370_TP_V.jpg_output_p5.jpg"
im6Name = ""
imgNames = [im1Name, im2Name, im3Name, im4Name, im5Name, im6Name]

tilePtn = 2

outName = "myfaceMoveLandmark_movelandmark.jpg"
outWidth = 600

imgSet = []
for name in imgNames:
    if len(name) != 0:
        im = cv2.imread(name)
    else:
        im = np.ones((outWidth, outWidth, 3),dtype=np.uint8)
        im *= 255
    imgSet.append(im)

# make tile img.
if tilePtn == 6:
    im_tile_resize = concat_tile_resize([[imgSet[0], imgSet[1]],
                                        [imgSet[2], imgSet[3]],
                                        [imgSet[4], imgSet[5]]])
elif tilePtn == 4:
    im_tile_resize = concat_tile_resize([[imgSet[0], imgSet[1]],
                                        [imgSet[2], imgSet[3]]])
elif tilePtn == 2:
    im_tile_resize = concat_tile_resize([[imgSet[0], imgSet[1]]])

w_tileImg = min([outWidth, im_tile_resize.shape[1]])
cv2.resize(im_tile_resize, (w_tileImg, int(im_tile_resize.shape[0] * w_tileImg / im.shape[1])), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(outName, im_tile_resize)