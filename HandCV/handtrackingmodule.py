import cv2 as cv
import mediapipe as mp
import time


class handDetector():
    def __init__(self,mode=False,maxHands=2, model_complexity=1,detectionCon=0.5,trackCon=0.5):
        # static_image_mode = False,
        # max_num_hands =2,
        # min_detection_confidence =0.5,
        # min_tracking_confidence = 0.5 taken reference from built in funtions code for parameters
        self.mode=mode
        self.maxHands =maxHands
        self.model_complexity = model_complexity
        self.detectionCon =detectionCon
        self.trackCon= trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.model_complexity,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw =True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)  
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks: 
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
            return img
                # draw the 21 coordintes on every hand encountered
            # for id, lm in enumerate(handLms.landmark):
            #     # print(id,lm)
            #     h, w, c = img.shape
            #     cx, cy = int(lm.x*w,), int(lm.y*h)

            #     print(id, cx, cy)
            #     if id == 4:
            #         cv.circle(img, (cx, cy), 15, (0, 255, 0), 5)

def main():
    pTime = 0
    cTime = 0  # previous an current time
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        if success:

            img = detector.findHands(img)
            cTime = time.time()
            fps = 1/(cTime - pTime)
            pTime = cTime


            cv.putText(img, str(int(fps)), (10, 70),cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), thickness=5)

            cv.imshow("Image", img)
            cv.waitKey(1)
        else:
            break


if __name__ == "__main__":
    main()