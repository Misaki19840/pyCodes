
import sys
import os
import dlib
import glob
import cv2



def drawparts(imgMat="",shape="",st="",ed=""):
    for id in range(st + 1, ed + 1):
        cv2.line(imgMat, (shape.part(id - 1).x, shape.part(id - 1).y), (shape.part(id).x, shape.part(id).y),
                 (255, 0, 0))
    # text_st = "st:" + str(st)
    # text_ed = "ed:" + str(ed)
    # cv2.putText(imgMat, text_st, (shape.part(st).x,shape.part(st).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
    # cv2.putText(imgMat, text_ed, (shape.part(ed).x, shape.part(ed).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

def savelandmark(predictor_path="",faces_folder_path=""):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        name, ext = os.path.splitext(f)
        txtName = name + ".txt"
        fp_w = open(txtName, mode='w')
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)

        imgMat = cv2.imread(f)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            shape = predictor(img, d)

            for ptId in range(shape.num_parts):
                #write landmark positions
                str_w = str(shape.part(ptId).x) + " " + str(shape.part(ptId).y) + "\n"
                fp_w.write(str_w)
                #draw landmarks
                cv2.circle(imgMat,(shape.part(ptId).x, shape.part(ptId).y),2,(255,0,0))

            drawparts(imgMat, shape, 0, 16) # face line
            drawparts(imgMat, shape, 36, 41) # leye
            drawparts(imgMat, shape, 17, 21)  # leyebrow
            drawparts(imgMat, shape, 42, 47) # reye
            drawparts(imgMat, shape, 22, 26) # reyebrow
            drawparts(imgMat, shape, 27, 30) # nose
            drawparts(imgMat, shape, 31, 35) # nose bottom
            drawparts(imgMat, shape, 48, 59) # mouth out
            drawparts(imgMat, shape, 60, 67) # mouth in

            fp_w.close()

            cv2.imshow("landmark", imgMat)
            saveimgname = name + "_landmarks.jpg"
            cv2.imwrite(saveimgname, imgMat)
            cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    faces_folder_path = "facefld"

    savelandmark(predictor_path,faces_folder_path)