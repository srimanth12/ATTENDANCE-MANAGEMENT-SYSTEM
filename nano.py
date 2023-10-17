# USAGE


# python detect_mask_video.py
import cv2
# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
print("##########imported from tensorflow.keras.applications.mobilenet_v2 import preprocess_input")
from tensorflow.keras.preprocessing.image import img_to_array
print("##########imported from tensorflow.keras.preprocessing.image import img_to_array")
from tensorflow.keras.models import load_model
print("######### imported from tensorflow.keras.models import load_model")
#from imutils.video import VideoStream

import numpy as np
import imutils
print("####################")
import time
import os
print("##############imported all")

print('###################################################33 NO problem/ with libraries importing ')



def gstreamer_pipeline(
    capture_width=320,
    capture_height=150,
    display_width=320,
    display_height=150,
    framerate=20,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
	    (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
	    confidence = detections[0, 0, i, 2]
	    if confidence > 0.5:
		    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		    (startX, startY, endX, endY) = box.astype("int")
		    (startX, startY) = (max(0, startX), max(0, startY))
		    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
		    face = frame[startY:endY, startX:endX]
		    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		    face = cv2.resize(face, (224, 224))
		    face = img_to_array(face)
		    face = preprocess_input(face)
		    face = np.expand_dims(face, axis=0)
		    faces.append(face)
		    locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
	    preds = maskNet.predict(faces)
    return (locs, preds)


print("[INFO] ################################################################################################################loading face detector model...")
prototxtPath = os.path.sep.join(['face_detector', "deploy.prototxt"])
weightsPath = os.path.sep.join(['face_detector',"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] #################################################################################################################loading face mask detector model...")
maskNet = load_model('mask_detector.model')

print("[INFO] ############################################################################################################### starting video stream...")
print(gstreamer_pipeline(flip_method=0))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if cap.isOpened():
    window_handle = cv2.namedWindow("mask-detect", cv2.WINDOW_AUTOSIZE)
    while cv2.getWindowProperty("mask-detect",0) >= 0:
        ret_val, frame = cap.read()
        frame = imutils.resize(frame, width=400)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.imshow("Frame", frame)
        keyCode = cv2.waitKey(30) & 0xFF
        if keyCode == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Unable to open camera")



'''
def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            frame = imutils.resize(img, width=400)
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.imshow("CSI Camera", frame)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    show_camera()


'''





