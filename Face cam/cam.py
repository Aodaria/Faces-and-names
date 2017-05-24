
import numpy as np
import cv2
import math
import sys

from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
import PIL.Image
import PIL.ImageChops


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


# normalize to compensate for exposure difference, this may be unnecessary
def compare_images(img1, img2):
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng


def frame2vect(face):
    #transforme l'image en vect
    try:
        face = cv2.resize(face,None,fx=200/len(face[0]), fy=200/len(face[1]), interpolation = cv2.INTER_CUBIC)
        a = face.flatten()
        #print len(a)
        return a
    except:
        pass

def reduceMat(mat):
    red = np.zeros((len(mat)/4, len(mat)/4))
    try:
        for i in range(len(mat)/4):
            for j in range(len(mat)/4):
                red[i,j] = (mat[i*4,j*4] + mat[i*4+1,j*4] + mat[i*4,j*4+1] + mat[i*4+1,j*4+1])
        return frame2vect(red)
    except:
        pass


def dist2vect(vect1, vect2):
    tmp = 0
    for a in range(len(vect1)):
        tmp += int((vect1[a] - vect2[a])) * int((vect1[a] - vect2[a]))
    print tmp

image = []

while(True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 2, 5)

    for (x,y,w,h) in faces:
        imagetmp = gray[y:y+h,x:x+w]
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):
        try:
            image = reduceMat(imagetmp)
        except:
            pass
    if cv2.waitKey(1) & 0xFF == ord('a'):
        dist2vect(reduceMat(imagetmp),image)


    cv2.imshow('frame',gray)


cap.release()
cv2.destroyAllWindows()
