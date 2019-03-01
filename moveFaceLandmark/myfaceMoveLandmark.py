#! /usr/bin/env python

import sys
import numpy as np
import cv2
import os
import copy

# Read points from text file
def readPoints(path):
    # Create an array of points.
    points = [];

    # Read points
    with open(path) as file:
        for line in file:
            if len(line) < 4:
                points.append((0,0))
                continue
            x, y = line.split()
            points.append((int(x), int(y)))

    return points


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst

# Warps triangular regions from img1 to img2
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]*((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect

def getFacialLandmarkImg(img = "", lpos = "", tset = "", color = (255, 0, 0)):
    imgshow = np.copy(img)
    for i in range(len(lpos)):
        cv2.circle(imgshow, lpos[i], 2,  color, -1
                   )
    for i in range(len(tset)):
        cv2.line(imgshow, tset[i][0], tset[i][1], color, 1)
        cv2.line(imgshow, tset[i][0], tset[i][2], color, 1)
        cv2.line(imgshow, tset[i][2], tset[i][1], color, 1)

    return imgshow

def moveLandmark(filename1 = "", offsetPx = -10):
    img1 = cv2.imread(filename1);
    img2 = cv2.imread(filename1);
    img1Warped = np.copy(img1);

    # Read array of corresponding points
    name, ext = os.path.splitext(filename1)
    points1 = readPoints(name + '.txt')

    # Read triangles from tri.txt
    dt = []
    with open("triface.txt") as file:
        for line in file:
            p1_idx, p2_idx, p3_idx = line.split()
            print("p1_idx, p2_idx, p3_idx : ", p1_idx, p2_idx, p3_idx )
            dt.append((int(p1_idx), int(p2_idx), int(p3_idx)))

    # set outside points
    # add new triangles
    pointsOut = []
    tsetOut = []
    # left face line
    for i in range(1, 6):
        xpos = points1[i][0] - 20
        ypos = points1[i][1]
        points1.append((int(xpos), int(ypos)))
        dt.append((i, i-1, len(points1) - 1))
        if i != 1:
            dt.append((i-1, len(points1) - 2, len(points1) - 1))
    # right face line
    for i in range(11, 16):
        xpos = points1[i][0] + 20
        ypos = points1[i][1]
        points1.append((int(xpos), int(ypos)))
        dt.append((i, i+1, len(points1) - 1))
        if i != 11:
            dt.append((i, len(points1) - 2, len(points1) - 1))

    #make offset point
    points2 = copy.deepcopy(points1)
    # left face line
    for i in range(1, 5):
        points2[i] = (points1[i][0] - offsetPx, points1[i][1])
    # right face line
    for i in range(12, 16):
        points2[i] = (points1[i][0] + offsetPx, points1[i][1])

    # Find delanauy traingulation for convex hull points
    sizeImg = img1.shape
    rect = (0, 0, sizeImg[1], sizeImg[0])

    # dt = calculateDelaunayTriangles(rect, hull2)
    tset1 = []
    tset2 = []

    for i in range(len(dt)):
        p1Idx, p2Idx, p3Idx = int(dt[i][0]), int(dt[i][1]), int(dt[i][2])
        tset1.append([points1[p1Idx], points1[p2Idx], points1[p3Idx]])
        tset2.append([points2[p1Idx], points2[p2Idx], points2[p3Idx]])

    if len(dt) == 0:
        quit()

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(tset1[i][j])
            t2.append(tset2[i][j])

        warpTriangle(img1, img1Warped, t1, t2)

    imgshow = getFacialLandmarkImg(img1, points1, tset1)
    cv2.imshow("img1 lamdmark", imgshow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    saveimgname = filename1 + "_beforewarp.jpg"
    cv2.imwrite(saveimgname, imgshow)

    imgshow = getFacialLandmarkImg(img1Warped, points2, tset2)
    cv2.imshow("img1 lamdmark warped", imgshow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    saveimgname = filename1 + "_afterwarp.jpg"
    cv2.imwrite(saveimgname, imgshow)

    imgsbase = getFacialLandmarkImg(img1, points1, [])
    imgshow = getFacialLandmarkImg(imgsbase, points2, tset2, (0,0,255))
    cv2.imshow("img1 lamdmark warped(before and after)", imgshow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    saveimgname = filename1 + "_beforeafterwarp.jpg"
    cv2.imwrite(saveimgname, imgshow)

    return img1Warped

if __name__ == '__main__':

    # Make sure OpenCV is version 3.0 or above
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        print >> sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher'
        sys.exit(1)

    # Read images
    filename1 = 'imgfld/ookawa918IMGL1370_TP_V.jpg'

    offset = 5
    output = moveLandmark(filename1, offset)

    cv2.imshow("Face landmark moved", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    strOffs = "_p" + str(abs(offset)) if (offset > 0) else "_m" + str(abs(offset))
    saveimgname = filename1 + "_output" + strOffs + ".jpg"
    cv2.imwrite(saveimgname, output)

    #showFacialLandmark("before", img1, points1, tset1)
    #showFacialLandmark("after", output, points2, tset2)

