import cv2
import numpy as np
import os 
import time 
from datetime import datetime
recognizer = cv2.face.LBPHFaceRecognizer_create()
cascadePath = "D:\ECE 2nd year sem1\AIML Project\Attendance\haarcascade_frontalface_default.xml"
recognizer.read('D:\ECE 2nd year sem1\AIML Project\Attendance\Trainner.yml')
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
#initiate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Srimanth', 'Praveen','Shrikar'] 
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
      
l=[]

def atten(x):
    roll = {'Praveen' : '21ECB0B31', 'Srimanth' : '21ECB0B33','Shrikar' : '21ECB0B21', 'unknown' : '0'}
    file = open('D:\ECE 2nd year sem1\AIML Project\Attendance\Attendance.csv', 'a')
    y = x.strip()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    # date = now.strftime("%x")
    # day = now.strftime("%A")
    if y not in l:
        #file.write('"{0}", "{1}", "{2}", "{3}", "{4}"'.format(roll[y], y, current_time, date, day))
        file.write('"{0}", "{1}", "{2}"'.format(roll[y], y,current_time))        
        file.write('\n')
    l.append(y)
                  
while True:
    
    num_frames=120
#Start time
    start = time.time()
 
#Grab a few frames
    for i in range (0, num_frames) :
        ret, frame = cam.read()
 
#End time
    end = time.time()
 
# Time elapsed
    seconds = end - start
    print ("Time taken : {0} seconds".format(seconds))
 
# Calculate frames per second
    fps  = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))
    ret, img =cam.read()
   #img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.3,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # If confidence is less them 100 ==> "0" : perfect match 
        #if (confidence < 100):
        #   id = names[id]
        #  confidence = "  {0}%".format(round(100 - confidence))
        #else:
        #    id = "unknown"
        #   confidence = "  {0}%".format(round(100 - confidence))
        if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(confidence))
        
        cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
        atten(id)            
        cv2.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )  
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()