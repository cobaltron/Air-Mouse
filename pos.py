import numpy as np
import dlib
import cv2
from imutils.video import WebcamVideoStream
import imutils
from scipy.spatial import distance as dist
from pynput.mouse import Button, Controller
import sys
import pyautogui
from pos_utils import *
pyautogui.FAILSAFE=False
FULL_POINTS = list(range(0, 68))
FACE_POINTS = list(range(17, 68))

RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

COUNTER_LEFT = 0
TOTAL_LEFT = 0

COUNTER_RIGHT = 0
TOTAL_RIGHT = 0
mouse=Controller()


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cap = WebcamVideoStream(src=1).start()
f=0
prev_cx = 200; prev_cy = 200;

while(True):
    image = cap.read()
    image=cv2.flip(image,1)
    cv2.imshow("Output",image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    h=image.shape[0]
    w=image.shape[1]
    scrn_h,scrn_w=1080,1920
    scale_x=int(scrn_w/w)
    scale_y=int(scrn_h/h)
    #print(scale_x)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):

        res_cx = []; res_cy = []
        #print(len(rects))
        for i in range(9):
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            p,img=pose(gray,shape[30],shape[8],shape[45],shape[36],shape[54],shape[48])
            #print("Rx={0}".format(rx)+"Ry={0}".format(ry))
            #print(p)
            cx=p[0]
            cy=p[1]
            res_cx.append(cx); res_cy.append(cy)
            #cv2.imshow("Output",img)
            

        
        #cx = int(np.array(res_cx).median()); cy= int(np.array(res_cy).median())
        cx = int(np.median(np.array(res_cx))) ; cy = int(np.median(np.array(res_cy)))

        m = abs(cx-prev_cx); n = abs(cy-prev_cy)
        #print('[',m,',', n,']')
        if ((m > 2) and (n > 2)):
        	#print('Changing...')
        	#mouse.position=((cx*scale_x),(cy*scale_y))
        	pyautogui.moveTo((cx*scale_x),(cy*scale_y),0.25, pyautogui.easeInQuad) 
        
        #mouse.position=((cx*scale_x),(cy*scale_y))
        prev_cx = cx; prev_cy = cy;

        left_eye = shape[LEFT_EYE_POINTS]
        right_eye = shape[RIGHT_EYE_POINTS]
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        if ear_left < EYE_AR_THRESH:
            COUNTER_LEFT += 1
        else:
            if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
                TOTAL_LEFT += 1
                mouse.press(Button.right)
                mouse.release(Button.right)
                pyautogui.click(button='right')
                print("Right eye blinked")
            COUNTER_LEFT = 0
        if ear_right < EYE_AR_THRESH:
            COUNTER_RIGHT += 1
        else:
            if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:
                TOTAL_RIGHT += 1
                mouse.press(Button.left)
                mouse.release(Button.left)
                pyautogui.click(button='left')
                print("Left eye blinked")
            COUNTER_RIGHT = 0
        #print(shape[0])
        '''
        cv2.imshow("Output",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            '''
cap.stop()
cv2.destroyAllWindows()
