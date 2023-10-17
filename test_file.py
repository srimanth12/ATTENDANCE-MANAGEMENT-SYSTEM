import torch
import  cv2 
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from torchvision import datasets 
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from imutils.video import FPS 

import numpy as np
import pandas as pd
import os
mtcnn = MTCNN(margin=20, keep_all=True, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval() 

# Load a single image and display
load_data = torch.load('/home/prasanjith/Downloads/data.pt') 
embedding_list = load_data[0] 
name_list = load_data[1] 

cam = cv2.VideoCapture('/home/prasanjith/Desktop/face_nitw/bean.mp4')
fps=FPS().start()

while True:
    ret, frame = cam.read()
    if not ret:
        print("fail to grab frame, try again")
        break
    #frame=cv2.resize(frame, (720,1280))
    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
    
    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)
                
        for i, prob in enumerate(prob_list):
            if prob>0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                
                dist_list = []
                box = boxes[i] # list of matched distances, minimum distance is used to identify the person
                
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list) # get minumum dist value
                min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                name = name_list[min_dist_idx] # get name corrosponding to minimum dist
                
                box = boxes[i] 
                
                original_frame = frame.copy() # storing copy of frame before drawing on it
                
                if min_dist<0.90:
                    frame = cv2.putText(frame, name+' '+str(min_dist), (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)
                
                frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (255,0,0), 2)

    cv2.imshow("IMG", frame)
        
    
    k = cv2.waitKey(1)
    if k%256==27: # ESC
        print('Esc pressed, closing...')
        break
    fps.update()
        
    '''elif k%256==32: # space to save image
        print('Enter your name :')
        name = input()
        
        # create directory if not exists
        if not os.path.exists('photos/'+name):
            os.mkdir('photos/'+name)
            
        img_name = "photos/{}/{}.jpg".format(name, int(time.time()))
        cv2.imwrite(img_name, original_frame)
        print(" saved: {}".format(img_name))'''
        
        
fps.stop() 
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("FPS:{:.2f}".format(fps.fps()))

cam.release()
cv2.destroyAllWindows()
