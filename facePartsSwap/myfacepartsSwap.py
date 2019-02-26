#! /usr/bin/env python

import sys
import numpy as np
import cv2
import os

# Read points from text file
def readPoints(path):
    # Create an array of points.
    points = [];

    # Read points
    with open(path) as file:
        for line in file:
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


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


# calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # create subdiv
    subdiv = cv2.Subdiv2D(rect);

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList();

    delaunayTri = []

    pt = []

    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            # Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
                        # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunayTri


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

def getFacialLandmarkImg(img = "", lpos = "", tset = ""):
    imgshow = np.copy(img)
    for i in range(len(lpos)):
        cv2.circle(imgshow, lpos[i], 2,  (255,0,0))
    for i in range(len(tset)):
        cv2.line(imgshow, tset[i][0], tset[i][1], (255,0,0), 1)
        cv2.line(imgshow, tset[i][0], tset[i][2], (255, 0, 0), 1)
        cv2.line(imgshow, tset[i][2], tset[i][1], (255, 0, 0), 1)

    return imgshow

def partsSwap(filename1 = "", filename2 = "", swLeye = 0, swReye = 0, swNose = 0, swMouth = 0):
    img1 = cv2.imread(filename1);
    img2 = cv2.imread(filename2); # base img
    img1Warped = np.copy(img2);

    partsfile = "tri_leye.txt"

    partslist = []
    if swLeye : partslist.append("tri_leye.txt")
    if swReye: partslist.append("tri_reye.txt")
    if swNose: partslist.append("tri_nose.txt")
    if swMouth: partslist.append("tri_mouth.txt")

    # get shape info
    if len(img2.shape) == 3:
        height, width, channels = img2.shape[:3]
    else:
        height, width = img2.shape[:2]
        channels = 1

    # Read array of corresponding points
    name, ext = os.path.splitext(filename1)
    points1 = readPoints(name + '.txt')
    name, ext = os.path.splitext(filename2)
    points2 = readPoints(name + '.txt')

    # dt = calculateDelaunayTriangles(rect, hull2)
    dt = []
    tset1 = []
    tset2 = []
    p_transform = []

    output = np.copy(img2);
    # Read triangles from tri.txt
    for partsfile in partslist:
        mask_parts = np.zeros(img2.shape, dtype=np.uint8)
        with open(partsfile) as file:
            for line in file:
                x, y, z = line.split()
                print("x, y, z: ", x, y, z )

                x = int(x)
                y = int(y)
                z = int(z)

                dt.append((x,y,z))
                tset1.append([points1[x], points1[y], points1[z]])
                tset2.append([points2[x], points2[y], points2[z]])

                p_transform.append(points2[x])
                p_transform.append(points2[y])
                p_transform.append(points2[z])

                tripos = []
                tripos.append(points2[x])
                tripos.append(points2[y])
                tripos.append(points2[z])

                # create small mask for each triangles
                mask_small = np.zeros(img2.shape, dtype=np.uint8)
                cv2.fillConvexPoly(mask_small, np.int32(tripos), (255,255,255))
                mask_parts |= mask_small

        if len(dt) == 0:
            quit()

        # Apply affine transformation to each triangles
        for i in range(0, len(dt)):
            t1 = []
            t2 = []

            # get points for img1, img2 corresponding to the triangles
            for j in range(0, 3):
                t1.append(tset1[i][j])
                t2.append(tset2[i][j])

            warpTriangle(img1, img1Warped, t1, t2)

        mask_c1, mask_c2, mask_c3 = cv2.split(mask_parts)
        maskMoments = cv2.moments(mask_c1)
        cx = int(maskMoments['m10'] / maskMoments['m00'])
        cy = int(maskMoments['m01'] / maskMoments['m00'])
        center = (int(cx), int(cy))

        # Clone seamlessly.
        output = cv2.seamlessClone(np.uint8(img1Warped), output, mask_parts, center, cv2.NORMAL_CLONE)

        img1land = getFacialLandmarkImg(img1, points1, tset1)
        cv2.imshow("img1 lamdmark", img1land)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        saveimgname = filename1 + "_beforewarp.jpg"
        cv2.imwrite(saveimgname, img1land)

        img1landWarp = getFacialLandmarkImg(img1Warped, points2, tset2)
        cv2.imshow("img1 warp lamdmark", img1landWarp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        saveimgname = filename1 + "_afterwarp.jpg"
        cv2.imwrite(saveimgname, img1landWarp)

    return output

if __name__ == '__main__':

    # Make sure OpenCV is version 3.0 or above
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        print >> sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher'
        sys.exit(1)

    # Read images
    filename1 = 'facefld/model10211041_TP_V4.jpg'
    filename2 = 'facefld/XEN7615021_TP_V.jpg'

    output = partsSwap(filename1, filename2, swLeye = 1, swReye = 1, swNose = 0, swMouth = 1)

    cv2.imshow("Face Swapped", output)
    saveimgname = filename2 + "_faceswapped.jpg"
    cv2.imwrite(saveimgname, output)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

