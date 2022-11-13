import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# features = np.load('features.npy')
# labels = np.load('labels.npy')

people = ['Boyka', 'Deborah', 'Jon', 'Maluma']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(
    r'C:\Users\momsk\OneDrive\Desktop\Krish\MLB\Opencv\Images\TEST\Jon\intro-1505083758.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

#Detect face in image
faces_rect = haar_cascade.detectMultiScale(gray,1.1,9
)

for(x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label,confidence = face_recognizer.predict(faces_roi)
    print(f'label = {people[label]} with a confidence of {confidence}')

    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img, (x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('img',img)
cv.waitKey(0)
    
