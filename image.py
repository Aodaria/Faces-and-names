#http://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html

from imutils.video import VideoStream
from matplotlib import pyplot as plt
from imutils import face_utils
import datetime
import numpy as np
import math
import argparse
import imutils
import time
import dlib
import cv2



def calcDist(a,b):
    return math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]))

def dist2vect(a,b):
    return math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]))


def findDist(shape, frame):

    valeur_morpho = []
    oo = calcDist(shape[0], shape[16])
    cv2.line(frame, (shape[0][0], shape[0][1]),(shape[16][0], shape[16][1]), (255,255,0))

    om = (calcDist(shape[0], shape[8]) + calcDist(shape[8], shape[16]) )/ 2

    egeg = calcDist(shape[36], shape[39])
    cv2.line(frame, (shape[36][0], shape[36][1]),(shape[39][0], shape[39][1]), (255,255,0))

    eded = calcDist(shape[42], shape[45])
    cv2.line(frame, (shape[42][0], shape[42][1]),(shape[45][0], shape[45][1]), (255,255,0))

    bgbd = calcDist(shape[48], shape[54])
    cv2.line(frame, (shape[48][0], shape[48][1]),(shape[54][0], shape[54][1]), (255,255,0))

    ngnd = calcDist(shape[31], shape[35])
    cv2.line(frame, (shape[31][0], shape[31][1]),(shape[35][0], shape[35][1]), (255,255,0))

    ogtomenton = calcDist(shape[0], shape[8])
    cv2.line(frame, (shape[0][0], shape[0][1]),(shape[8][0], shape[8][1]), (255,255,0))

    odtomenton = calcDist(shape[8], shape[16])
    cv2.line(frame, (shape[8][0], shape[8][1]),(shape[16][0], shape[16][1]), (255,255,0))


    valeur_morpho.append(oo)
    valeur_morpho.append(om)
    valeur_morpho.append(egeg)
    valeur_morpho.append(eded)
    valeur_morpho.append(bgbd)
    valeur_morpho.append(ngnd)
    valeur_morpho.append(ogtomenton)
    valeur_morpho.append(odtomenton)

    return  valeur_morpho



def distPoint(a,b):
    return float(a)-float(b)

def ecrire(val, name):
    output = open('fichiertet.txt', 'a')
    for i in val:
        output.write(str(i) + ";")
    output.write(str(name) + ";\n")
    output.close()

def longueurVect(vect):
    tmp = 0
    for i in vect:
        tmp += float(i)*float(i)
    return math.sqrt(tmp)

def dist2vectLocal(vect1, vect2):
    tmp = []
    for i in range(len(vect1)):
        tmp.append(distPoint(vect1[i], vect2[i]))
    return tmp


def verification(val):
    info = open('fichiertet.txt','r')
    for line in info.readlines():
        line = line.split(";")
        tmp = []
        for i in line:
            try:
                tmp.append(float(i))
            except:
                pass

        longface = longueurVect(dist2vectLocal(val, tmp))
        print longface

        if longface < 1:
            print 'I know you !'
            return "X"
        else:
            print 'You\'re unknown...'
            return "xx"


def preparerImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 2, 5)
    imagetmp = []

    for (x,y,w,h) in faces:
        imagetmp = gray[y:y+h,x:x+w]
    imagetmp = np.array(imagetmp, dtype = np.uint8)
    imagetmp = imutils.resize(imagetmp, width=500)

    return imagetmp


def colorHair(frame):
    #A refaire car chauve qui peut, ca marche pas
    #importer le code auxiliaire avec histogramme local
    #
    rects = detector(frame, 0)
    for rect in rects:
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)

    pos = [shape[22], shape[25]]

    centre = [int(pos[0][0] + pos[1][0])/2, int(pos[0][1] + pos[1][1])/2]
    col = frame[centre[0], centre[1]]
    col = [col[0],col[1],col[2]]
    chatin = dist2vect([0,0,0], col)
    roux = dist2vect([176,85,10], col)
    blond = dist2vect([176,165,110], col)

    if chatin < roux:
        if chatin < blond:
            color = "chatin"
        else:
            color = "blond"
    else:
        if roux < blond:
            color = "roux"
        else:
            color = "blond"
    print col
    cv2.circle(frame, (centre[0], centre[1]), 3, (123, 0, 0), -1)

    return color



def acquisition(image):
    imagetmp = preparerImage(image)
    rects = detector(imagetmp, 0)

    for rect in rects:
        shape = predictor(imagetmp, rect)
        shape = face_utils.shape_to_np(shape)
        tmp = findDist(shape, imagetmp)
        a = verification(tmp)
        ecrire(tmp, a)
        print colorHair(image) #ne marche pas
        for (x, y) in shape:
            cv2.circle(imagetmp, (x, y), 1, (0, 0, 0), -1)

    cv2.imshow("1", imagetmp)
    cv2.imshow("Frame", image)

"""
####### Core options
"""


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


image = cv2.imread('man.png', 1)

acquisition(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
