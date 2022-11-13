import cv2 as cv
import numpy as np

#haar_cascade is very sensitive to noise to detect more facces we will have to change the minNeighbours to a lesser value so that more noise is included and more faces are detected 

img = cv.imread('Images/group2.jpg')
cv.imshow('Maluma', img)

# detection happens by identifying the edges in the image so color image is not required for face detection
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

# reading the haarcascade files
haar_cascade = cv.CascadeClassifier('haar_face.xml')
# last parameter signifies the minimum number of neighbours a rectangle should have to be called a face
faces_rect = haar_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=1)
# detectMultiScale  will take in am image and other variables to detect a face and return the rectangluar coordinates as a list to faces_rect
print(f'Number of faces found = {len(faces_rect)}')
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('faces detect', img)
cv.waitKey(0)
#for extending to videos we have to apply aar_cascades to every frame