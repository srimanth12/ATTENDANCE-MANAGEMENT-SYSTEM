import cv2
import numpy as np

thres = 0.50  # Threshold to detect object
nms_threshold = 0.30
cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    print(classNames)
configPath = 'ssd_mobilenet.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
#print(weightsPath)
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
#cap = cv2.VideoCapture(1)
#cap.set(3,1280)
#cap.set(4,720)
#cap.set(10,150)
from time import time
t0 = time()
while True:
        '''cap = cv2.VideoCapture(1)
        cap.set(3,640)
        cap.set(4,480)
        classNames = []
        classFile = 'coco.names'
        with open(classFile,'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')
            #print(classNames)
        configPath = 'ssd_mobilenet.pbtxt'
        weightsPath = 'frozen_inference_graph.pb'
        #print(weightsPath)
        net = cv2.dnn_DetectionModel(weightsPath, configPath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)#cap = cv2.VideoCapture(1)
        #cap.set(3,1280)
        #cap.set(4,720)
        # cap.set(10,150)
        #cap.set(3,640)
        #cap.set(4,480)'''
        success, img = cap.read()
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
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        p=(time() - t0)
        print(f"done in {p}s.)" )#% (time() - t0))
        cv2.imshow("Output", img)
        cv2.waitKey(100000)
cap.release()
