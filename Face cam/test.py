import sys
import os
import dlib
import glob
import numpy as np
from skimage import io
import cv2


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

ii = 1
while (ii    == 1):

    ret, frame = cap.read()
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
    dets = detector(frame, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))
        cv2.rectangle(frame,(d.left(),d.top()),(d.right(), d.bottom()),(0,255,0),3)
    win.clear_overlay()
    #win.set_image(frame)
    #win.add_overlay(dets)
    cv2.imshow('frame',frame)
#    dlib.hit_enter_to_continue()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finally, if you really want to you can ask the detector to tell you the score
# for each detection.  The score is bigger for more confident detections.
# The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
# Also, the idx tells you which of the face sub-detectors matched.  This can be
# used to broadly identify faces in different orientations.


"""
cap = cv2.VideoCapture(0)


predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]
#faces_folder_path = sys.argv[3]

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
#facerec = dlib.face_recognition_model_v1(face_rec_model_path)

win = dlib.image_window()

while (true):

    ret, frame = cap.read()
    # Now process all the images
    #print("Processing file: {}".format(f))
    #img = io.imread(f)

    win.clear_overlay()
    win.set_image(frame)

    dets = detector(frame, 1)


    cv2.imshow('frame',dets)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""
