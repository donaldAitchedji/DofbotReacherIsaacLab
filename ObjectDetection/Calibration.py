import cv2 as cv
import numpy as np


def nothing(x):
    pass

cv.namedWindow("Trackbars", cv.WINDOW_NORMAL)
cv.resizeWindow("Trackbars", 400, 300)


cv.createTrackbar("H_min","Trackbars",40,179,nothing)
cv.createTrackbar("H_max","Trackbars",80,179,nothing)
cv.createTrackbar("S_min","Trackbars",70,255,nothing)
cv.createTrackbar("S_max","Trackbars",200,255,nothing)
cv.createTrackbar("V_min","Trackbars",30,255,nothing)
cv.createTrackbar("V_max","Trackbars",255,255,nothing)



def segmentationColor(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_rgb=cv.cvtColor(hsv,cv.COLOR_BGR2RGB)

    cv.imshow("hsv", hsv_rgb)
    h_min = cv.getTrackbarPos("H_min","Trackbars")
    h_max = cv.getTrackbarPos("H_max","Trackbars")

    s_min = cv.getTrackbarPos("S_min","Trackbars")
    s_max = cv.getTrackbarPos("S_max","Trackbars")

    v_min = cv.getTrackbarPos("V_min","Trackbars")
    v_max = cv.getTrackbarPos("V_max","Trackbars")

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    maskColor = cv.inRange(hsv, lower, upper)
    
    mask = maskColor 

    result = cv.bitwise_and(frame, frame, mask = mask)
   
    cv.imshow('frame', frame)
    cv.imshow('maskColor', maskColor)
    
    cv.imshow('Result',result)


img=cv.imread('Photos/imagesProjetLicence/imageSegmentationCouleur_0000.jpg')
#cv.imshow('Image_0006',img)

while True:

    segmentationColor(img)

    if cv.waitKey(1) & 0xFF == ord("d"):
        break

cv.destroyAllWindows()

