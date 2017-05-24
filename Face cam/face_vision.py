from imutils import face_utils
import numpy as np
import math
import argparse
import imutils
import dlib
import cv2


"""
Chargement des parametres
"""
def calcDist(a,b):
    return math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]))

def findDist(mat):
    oo = calcDist(shape[0], shape[16])
    om = (calcDist(shape[0], shape[8]) + calcDist(shape[8], shape[16]) )/ 2
    egeg = calcDist(shape[36], shape[39])
    eded = calcDist(shape[42], shape[45])
    bgbd = calcDist(shape[48], shape[54])
    ngnd = calcDist(shape[31], shape[35])

    return (oo, om, egeg, eded, bgbd, ngnd)

def dist2vect(vect1, vect2):
    tmp = 0
    for a in range(len(vect1)):
        tmp += int((vect1[a] - vect2[a])) * int((vect1[a] - vect2[a]))
    print tmp


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")

args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

image = cv2.imread('pic.jpg')
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)


    for i in range(0, 16):
        cv2.line(image,(shape[i][0],shape[i][1]),(shape[i+1][0], shape[i+1][1]), (255,255,0), 2)


    print findDist(shape)

    cv2.imshow('gray',image)

cv2.waitKey(0)
