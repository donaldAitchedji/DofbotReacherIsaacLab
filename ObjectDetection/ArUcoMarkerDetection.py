import cv2 as cv
import cv2.aruco as aruco


# dictionnaire de marqueurs
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

# paramètres de détection
parameters = aruco.DetectorParameters()

detector = aruco.ArucoDetector(aruco_dict, parameters)



cap = cv.VideoCapture(0) 

while True:
    isTrue,frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #gray = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
    #cv.imshow("grayBlur",gray)
    # détection des marqueurs
    corners, ids, rejected = detector.detectMarkers(gray)
    print(ids)
    if ids is not None:

        # dessiner les marqueurs détectés
        aruco.drawDetectedMarkers(frame, corners, ids)
        
    if cv.waitKey(20) & 0xFF==ord('q'):
        break
    cv.imshow('Rect Area',frame)

cap.release()
cv.destroyAllWindows()

#frame=cv.imread('Photos/imagesProjetLicence/imageAruco1_0004.jpg')

