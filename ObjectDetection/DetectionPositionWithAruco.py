import cv2 as cv
import numpy as np
import cv2.aruco as aruco

width_img = 640
height_img = 480

# Paramètres intrinsec de la camera
fx, fy = 800, 800
cx, cy = 320, 240

# --- Matrice intrinsèque caméra ---
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

# Coefficients de distorsion
dist_coeffs = np.zeros((4, 1))

#Vecteur de translation de la transformation de repère de la camera à la base 
T_cb = np.array([0.002, 0.139, 0.153])

#Matrice de rotation de la transformation de repère de la caméra à la base 
R_cb = np.array([[ 0.99983586,-0.01807169 ,  0.00129312],
               [-0.01734706, -0.93425894 , 0.35617316],
               [-0.00522854, -0.35613713, -0.93441908]])



# dictionnaire de marqueurs
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

# paramètres de détection
parameters = aruco.DetectorParameters()

detector = aruco.ArucoDetector(aruco_dict, parameters)


cube_edge_size = 0.03  # Taille réelle du marqueur en mètres



def euclide_distance(ref,estim):
    return np.sqrt((ref[0]-estim[0])**2+(ref[1]-estim[1])**2+(ref[2]-estim[2])**2)

def position_from_cam_to_base(pos_cam):
    return R_cb@pos_cam + T_cb



def  solvePnPAruco(corners):
    # Nouveau bloc avec 4 points pour solvePnP
    corner = corners[0].reshape(4, 2)  # Reshape pour obtenir les coins individuels
    img_points = np.array([
        corner[0],  # coin supérieur gauche
        corner[1],  # coin supérieur droit
        corner[2],  # coin inférieur droit
        corner[3]   # coin inférieur gauche
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


#Reading video with webcam and applying Canny edge detection
cap = cv.VideoCapture(1) 
cap.set(cv.CAP_PROP_FRAME_WIDTH, width_img)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height_img)
cap.set(cv.CAP_PROP_FPS, 30)


while True:
    isTrue,frame = cap.read()
    
    # Converting to grayscale
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    # détection des marqueurs
    corners, ids, rejected = detector.detectMarkers(gray)
    print(ids)
    if ids is not None:

        # dessiner les marqueurs détectés
        aruco.drawDetectedMarkers(frame, corners, ids)

        solved = solvePnPAruco(corners)
        center_cam = solved[0]
        print(center_cam)
        center_base = position_from_cam_to_base(center_cam)
        print(center_base)
        
        cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, solved[1], solved[2], 3)

        
    cv.imshow('Gray',frame)

    if cv.waitKey(20) & 0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
