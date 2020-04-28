# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:12:18 2020

@author: ADMIN
"""

import cv2 as cv
import numpy as np
from imutils import perspective
from scipy.spatial import distance as dist

orgImg = cv.imread("1.jpeg")
img = orgImg.copy()
cv.imshow("Image",img)
grayScale = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("GrayScaleImage",grayScale)
ret,thresh = cv.threshold(grayScale,127,255,cv.THRESH_BINARY) 
cv.imshow("Thresh Image",thresh)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv.contourArea(contour) > 100:
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect)
        box = np.array(box, dtype = "int")
        print(box)
        box = perspective.order_points(box)
        print(box)
        (topLeft, topRight, bottomRight, bottomLeft) = box
        height = dist.euclidean(topLeft,bottomLeft)
        width = dist.euclidean(topLeft,topRight)
        print("Height:{} Width:{}".format(height,width))
cv.drawContours(orgImg, contours, -1, (0,255,0), 3)
cv.imshow("contours",orgImg)

cv.waitKey(0)
cv.destroyAllWindows()
