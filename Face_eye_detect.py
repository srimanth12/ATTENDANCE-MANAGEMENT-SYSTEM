import cv2
face_cascade = cv2.CascadeClassifier("D:\ECE 2nd year sem1\AIML Project\Attendance\haarcascade_frontalface_default.xml") 
  
eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")  

# # capture frames from a camera 
cap = cv2.VideoCapture(0) 
# import time 
# # loop runs if capturing has been initialized. 
while 1:  

#     num_frames=120
# #Start time
#     start = time.time()
 
# #Grab a few frames
#     for i in range (0, num_frames) :
#         ret, frame = cap.read()
 
# #End time
#     end = time.time()
 
# # Time elapsed
#     seconds = end - start
#     print ("Time taken : {0} seconds".format(seconds))
 
# # Calculate frames per second
#     fps  = num_frames / seconds
#     print("Estimated frames per second : {0}".format(fps))
    # reads frames from a camera 
    ret, img = cap.read()
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
  
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
  
        # Detects eyes of different sizes in the input image 
        eyes = eye_cascade.detectMultiScale(roi_gray)  
  
        #To draw a rectangle in eyes

        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
  
    # Display an image in a window 
    cv2.imshow('img',img) 
  
    # Wait for Esc key to stop 
    k = cv2.waitKey(5)
    if k == 27: 
        break
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  
