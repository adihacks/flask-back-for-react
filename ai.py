from flask import Flask, Response
import cv2
from flask_cors import CORS

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

app = Flask(__name__)
CORS(app)

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")
offset = 20
imgSize = 400
folder = "Data/D"
counter = 0
labels = ["A","B","C","D","E","F","G","H","I","J"]


def video(imgOutput):

    # Open the webcam
    cap = cv2.VideoCapture(0)
    while True:
     img = cap.read()
     imgOutput = img.copy()
     hands, img  = detector.findHands(img)
     if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        imgWhite[0:imgCropShape[0],0:imgCropShape[1]] = imgCrop


        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(labels[index])


        else:
            k = imgSize / (w)
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        cv2.putText(imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

     cv2.imshow("Image", imgOutput)
     cv2.waitKey(1)
     return imgOutput
      

@app.route('/video')
def video():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Set up the response object
    def gen():
        # Read the frames from the webcam
        while True:
            ret, imgOutput = cap.read()
            if ret:
                 imgOutput = video(imgOutput)
            if not ret:
                break
            # Encode the frame as JPEG and yield it
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', imgOutput)[1].tobytes() + b'\r\n')

    # Set the response headers and return the response
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)
