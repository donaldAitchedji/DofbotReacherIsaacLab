import cv2 as cv
import numpy as np

# Paramètres de la caméra
width_img = 640
height_img = 480

# Paramètres intrinsec de la camera
fx, fy = 800, 800
cx, cy = 320, 240

# Matrice intrinsèque caméra 
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

# Coefficients de distorsion
dist_coeffs = np.zeros((4, 1))


# Taille réelle des coté du cube en mètres
cube_edge_size = 0.03

#Vecteur de translation de la transformation de repère de la camera à la base 
T_cb = np.array([0.002, 0.139, 0.153])

#Matrice de rotation de la transformation de repère de la caméra à la base 
R_cb = np.array([[ 0.99983586,-0.01807169 ,  0.00129312],
               [-0.01734706, -0.93425894 , 0.35617316],
               [-0.00522854, -0.35613713, -0.93441908]])


def position_from_cam_to_base(pos_cam):
    return R_cb@pos_cam + T_cb


def solvePnPCustom(box):
    # Nouveau bloc avec 4 points pour solvePnP
    img_points = np.array([
        box[0],  # coin supérieur gauche
        box[1],  # coin supérieur droit
        box[2],  # coin inférieur droit
        box[3]   # coin inférieur gauche
    ], dtype=np.float32)

    # Points 3D correspondants
    obj_points = np.array([
        [0, 0, 0],
        [cube_edge_size, 0, 0],
        [cube_edge_size, cube_edge_size, 0],
        [0, cube_edge_size, 0]
    ], dtype=np.float32)

    success, rvec, tvec = cv.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
    R, _ = cv.Rodrigues(rvec)
    center_obj= np.array([[cube_edge_size/2, cube_edge_size/2, cube_edge_size/2]], dtype=np.float32)
    center_cam = R @ center_obj.T + tvec
    center_cam = center_cam.flatten()

    if not success:
        return None

    return [ center_cam , rvec , tvec ]


def determinSeuil(gray):
    min = 255
    max = 0   
    for j in range( gray.shape[0] ) :
       for i in range( gray.shape[1] ) :
            if gray[j,i] > max :
                max = gray[j,i]
            if min > gray[j,i] :
                min = gray[j,i]
    mediane = (min+max)//2
    return mediane


def seuillage(gray , seuil_gris):
    img = gray.copy()
    for j in range(gray.shape[0]):
       for i in  range(gray.shape[1]):
            if gray[j,i] >= seuil_gris:
                img[j,i] = 255
            else:
               img[j,i] = 0
    return img


def segmentationColor(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    h_min1 = 0
    h_max1 = 20  #15
    
    h_min2 = 170
    h_max2 = 179  

    s_min1 = 24
    s_max1 = 255

    v_min1 = 194
    v_max1 = 255

    lower1 = np.array([h_min1 , s_min1 , v_min1])
    upper1 = np.array([h_max1 , s_max1 , v_max1])
    lower2 = np.array([h_min2 , s_min1 , v_min1])
    upper2 = np.array([h_max2 , s_max1 , v_max1])
    maskColor1 = cv.inRange(hsv, lower1, upper1)
    maskColor2 = cv.inRange(hsv, lower2, upper2)

    maskColor = cv.bitwise_or(maskColor1, maskColor2)
    result = cv.bitwise_and(frame, frame, mask = maskColor)

    cv.imshow('maskColor1', maskColor1)
    cv.imshow('maskColor2', maskColor2)
    #cv.imshow('maskColor', maskColor)
    
    cv.imshow('Result', result)
    
    return result

#Lecture de la vidéo
cap = cv.VideoCapture(1) 
cap.set(cv.CAP_PROP_FRAME_WIDTH, width_img)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height_img)
cap.set(cv.CAP_PROP_FPS, 30)

object_decteted = True

while True:
    isTrue,frame = cap.read()
    if not isTrue:
        break
    
    results = segmentationColor(frame)

    # Conversion en niveaux de gris
    gray = cv.cvtColor(results,cv.COLOR_BGR2GRAY)

    #Appliquer un flou gaussien pour rendre l'image moins floue
    blur = cv.GaussianBlur(gray,(7,7),cv.BORDER_DEFAULT)
    #cv.imshow('Blur',blur)

    #Appliquer un seuillage pour que les contours de l'objet soient plus facilement distinguables par le canny
    seuil_gris = determinSeuil(blur)
    blur_seuille = seuillage(blur , seuil_gris)
    
    #Traitement de l'image
    kernel = np.ones((3, 3), np.uint8)
    blur_seuille = cv.dilate(blur_seuille, kernel, iterations=4)  
    blur_seuille = cv.erode(blur_seuille, kernel, iterations=4)  
    #cv.imshow('Blur_seuille',blur_seuille)

    #Edge cascade
    canny = cv.Canny(blur_seuille,50,125)
    cv.imshow('Canny Edges',canny)

    #Determination des contours
    contours, hierarchy = cv.findContours(
        canny,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if contours is not None and contours != () :        
        #Déterminer le contour de l'objet en supposant que c'est le plus grand contour
        main_contour = max(contours, key = cv.contourArea)
        rect = cv.minAreaRect(main_contour)

        #Vérifier si c'est effectivement l'objet recherché qui a été détecté
        (_,_),(w,h),_ = rect
        print( "w : ",w," h : ",h)
        if w < 40 or h < 40 :
            print("NO OBJECT")
            object_decteted = False
    

        #Déterminer la position de l'objet dans le repère de la caméra et dans le repère de la base
        if object_decteted:
            box = cv.boxPoints(rect)   # 4 sommets
            box = box.astype(int)
            solved = solvePnPCustom(box)
            center_cam = solved[0]
            print(center_cam)
            center_base = position_from_cam_to_base(center_cam)
            print(center_base)
            cv.circle(frame, (box[0][0],box[0][1]),5, (255,0,0), -1) #BLUE
            cv.circle(frame, (box[1][0],box[1][1]),5, (0,0,255),-1) #RED
            cv.circle(frame, (box[2][0],box[2][1]),5, (255,255,255),-1) #white
            cv.circle(frame, (box[3][0],box[3][1]), 5,(0,0,0), -1) #black
            cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, solved[1], solved[2], 3)
            cv.drawContours(frame, [box], 0, (0, 255, 0), 2)

    cv.imshow('Rect Area',frame)
    object_decteted = True
    
    if cv.waitKey(20) & 0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
