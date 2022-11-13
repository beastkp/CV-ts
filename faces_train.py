from cProfile import label
import os
from tkinter.ttk import Style
import numpy as np
import cv2 as cv

people = ['Boyka', 'Deborah', 'Jon', 'Maluma']
DIR = r'C:\Users\momsk\OneDrive\Desktop\Krish\MLB\Opencv\Images\Training'

haar_cascade = cv.CascadeClassifier('haar_face.xml')
features =[]
labels = [] # we will set this in numeric values to reduce strain on PC while finding the images of that person, suppose boyka is 0 and maluma is 3 

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img);

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
# p=[]
# for i in os.listdir(r'C:\Users\momsk\OneDrive\Desktop\Krish\MLB\Opencv\Images\Training'):
#     p.append(i)
# print(p)
create_train()
print("Training done ---------------------")
# print(f'The length of features = {len(features)}')
# print(f'The length of labels = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)
face_recognizer = cv.face.LBPHFaceRecognizer_create()

#train the Recogniser on the features listand the labels list
face_recognizer.train(features,labels)
face_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)