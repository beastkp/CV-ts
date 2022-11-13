import cv2 as cv    
import mediapipe as mp
import time 

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands # mandatory to use for using this module
hands = mpHands.Hands()
# in Hands() in static_image_mode is defaultly false means it will detect or track depending on confidence level
mpDraw = mp.solutions.drawing_utils # method provided by mediapipe to draw a line on hand 

pTime =0
cTime =0 # previous an current time

while True:
    success, img =  cap.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB) # ew have to convert to RGB because this only uses RGB data
    results = hands.process(imgRGB) # calling the hands object to process the each frame using process method 
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: #handlms refers to a single hand in video
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm) 
                h, w, c= img.shape # height,wigth and channels 
                # position wrt to screen size
                cx, cy = int(lm.x*w,), int(lm.y*h)
                print(id, cx,cy) #returns the pixel wise coordinates of each point 
                if id==4:
                    cv.circle(img,(cx,cy),15,(0,255,0),5)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS) 
            # draw the 21 coordintes on every hand encountered 
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,3,(0,255,0),thickness=5)

    cv.imshow("Image",img)
    cv.waitKey(1) 