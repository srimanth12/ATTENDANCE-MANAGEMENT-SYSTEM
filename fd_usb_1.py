import cv2
print(cv2.__version__)
# Load the cascade
from time import time
t0 = time()
face_cascade = cv2.CascadeClassifier('/home/mllab/Desktop/haarcascade_frontalface_default.xml')
import numpy as np
thres = 0.50  # Threshold to detect object
nms_threshold = 0.30
cap = cv2.VideoCapture(1)
classNames = []
classFile = 'coco_names.txt'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    #print(classNames)
'''success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        bbox = list(bbox)
        #print(classIds,bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))
        # print(type(confs[0]))
        # print(confs)
        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
        print(indices)
        #if len(classIds)!=0:
        for classId,confidence,box in zip(classIds, confs, bbox):
            # for i in indices:
            #     i = i[0]
            #     box = bbox[1]
            # x, y, w, h = box[0], box[1], box[2], box[3]
            # cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
            # cv2.putText(img, classNames[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
            #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] +150 , box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)'''
'''configPath = 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
#print(weightsPath)
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
cap = cv2.VideoCapture(1)
# To capture video from webcam. 
#cap = cv2.VideoCapture(1)
# To use a video file as input 
#cap = cv2.VideoCapture('filename.mp4')'''
'''# Read the frame
            _, img = cap.read()
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img,"Number of faces detected= "+str(len(faces)),         (50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            # Display
            '''
while True:    
            t0 = time()  
        # Read the frame
            _, img = cap.read()
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img,"Number of faces detected= "+str(len(faces)),         (50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            # Display
            print("done in %0.3fs." % (time() - t0))
            cv2.imshow('img', img)
            # Stop if escape key is pressed
            k = cv2.waitKey(30) & 0xff
            if k==27:
              break
# Release the VideoCapture object
cap.release()
