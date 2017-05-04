from imutils.video import VideoStream
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
    valeur_morpho.append(odtomenton/eded)
    valeur_morpho.append(ogtomenton/egeg)


    return  valeur_morpho



def distPoint(a,b):
    return float(a)-float(b)

def ecrire(val):
    output = open('fichiertet.txt', 'a')
    for i in val:
        output.write(str(i) + ";")
    output.write("\n")
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

        if longface < 10:
            print 'I know you !'
            return 'name'
        else:
##            print 'You\'re unknown...'
            return '?'


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




def acquisition(image):
    imagetmp = preparerImage(image)
    rects = detector(imagetmp, 0)

    for rect in rects:
        shape = predictor(imagetmp, rect)
        shape = face_utils.shape_to_np(shape)
        tmp = findDist(shape, imagetmp)
        a = verification(tmp)

        ecrire(tmp)
        for (x, y) in shape:
            cv2.circle(imagetmp, (x, y), 1, (0, 0, 0), -1)

    cv2.imshow("1", imagetmp)
    cv2.imshow("Frame", image)



ap = argparse.ArgumentParser()
ap.add_argument("-r", "--picamera", type=int, default=-1,
    help="A camera need to be used")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while True:

    frame = vs.read()

    try:
        acquisition(frame)
    except:
        print "[X] Failed"


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        exit()
        break


    cv2.imshow("Frame", frame)

cv2.destroyAllWindows()
vs.stop()
