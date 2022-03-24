import os
import cv2
import time
import base64
import imutils
import argparse
import requests
import numpy as np
from datetime import datetime
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Global variable to run first time and to check value on subsequent runs
start = 0.0


def detect_and_predict_mask(current_frame, faceNet, maskNet):
    # dimensions of the current_frame to build a blob
    (h, w) = current_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(current_frame, 1, (300, 300), (104, 177, 123))

    # pass the blob within net to get face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # list of faces, their locations and list of predictions from the face mask network
    faces = []
    locations = []
    predictions = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidenceVal (probability)
        confidenceVal = detections[0, 0, i, 2]

        # check and get weak detections (confidenceVal should be greater than minimum confidenceVal)
        if confidenceVal > args["confidenceVal"]:
            # compute the (x, y)-coordinates of the bounding detectedBox for the object
            detectedBox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = detectedBox.astype("int")

            # making sure that bounding detectedBoxes are falling within dimensions of current_frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # preprocessing
            face = current_frame[startY:endY, startX:endX]  # extracting the face ROI
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)    # converting it from BGR to RGB
            face = cv2.resize(face, (224, 224))             # resizing it to 224x224
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)  # adding faces to their respective lists
            locations.append((startX, startY, endX, endY))  # adding detectedBoxes to their respective lists

    # prediction as if at least one face was detected
    if len(faces) > 0:
        # we make batch predictions on all faces at the same time in this loop
        faces = np.array(faces, dtype="float32")
        predictions = maskNet.predict(faces, batch_size=32)

    return locations, predictions


def convertForPost(current_frame):
    # encoding detected image to post on database
    cv2.imwrite('detected picture.jpg', current_frame)
    with open(r'/Users/user/PycharmProjects/pythonProject7/detected picture.jpg', 'rb') as file:
        encodedImage = base64.b64encode(file.read())

    return encodedImage


def postToWeb(current_frame):
    # posting data of current_frame to database
    detectedPicture = convertForPost(current_frame)
    # datetime object containing current date and time
    now = datetime.now()
    date = now.strftime("%d/%m/%Y")  # splitting date from datetime
    time = now.strftime("%H:%M:%S")  # splitting time from datetime

    url = 'https://hospital-manager.xyz/facemask/API.php'
    payload = {'date': date, 'time': time, 'image': detectedPicture}

    x = requests.post(url, data=payload)                    # posting


# building the argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="face_detector", help="face detector model directory path")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="trained face mask detector model path")
ap.add_argument("-c", "--confidenceVal", type=float, default=0.5, help="minimum probability for filtering")
args = vars(ap.parse_args())


# 'face detector model' is loading
print("[INFO] loading face detector model....")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# 'face mask detector model' is loading
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# streaming video by camera
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(5.0)

# looping over current_frames
while True:
    current_frame = vs.read()
    current_frame = imutils.resize(current_frame, width=700)

    # detecting faces in the current_frame then predicting if they are masked or not
    (locations, predictions) = detect_and_predict_mask(current_frame, faceNet, maskNet)

    # looping over the detected face locations
    for (detectedBox, pred) in zip(locations, predictions):
        (startX, startY, endX, endY) = detectedBox  # unpacking the bounding detectedBox and predictions
        (mask, withoutMask) = pred

        # class label and color for bounding detectedBox and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (255, 0, 0) if label == "Mask" else (0, 0, 255)
        decision = label

        # attaching probability
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # displaying label and bounding detectedBox
        cv2.putText(current_frame, label, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
        cv2.rectangle(current_frame, (startX, startY), (endX, endY), color, 2)

        if (decision == "No Mask"):
            if (start == 0 or time.time() - start > 30):
                postToWeb(current_frame)
                start = time.time()

    # the output current_frame
    cv2.imshow("current_frame", current_frame)
    key = cv2.waitKey(1)
