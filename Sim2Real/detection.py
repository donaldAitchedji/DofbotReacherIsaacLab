import cv2 as cv
import numpy as np
import cv2.aruco as aruco

class WasteDetector:
    def __init__(self, arucodict=aruco.DICT_4X4_100):

        #dictionnaire de marqueurs ArUco
        self.arucodict = aruco.getPredefinedDictionary(arucodict)

        # paramètres intrinsèques de la caméra
        self.camera_matrix = np.array([[800, 0, 320],
                                       [0, 800, 240],
                                       [0,   0,   1]
                                    ], dtype=np.float32)
        
        # paramètres de distorsion de la caméra
        self.dist_coeffs = np.zeros((4, 1))

        self.cube_size = 0.03
        
        #Vecteur de translation de la transformation de repère de la camera à la base 
        self.T_cb = np.array([0.002, 0.139, 0.153])

        #Matrice de rotation de la transformation de repère de la caméra à la base 
        self.R_cb = np.array([[ 0.99983586, -0.01807169,  0.00129312],
                              [-0.01734706, -0.93425894,  0.35617316],
                              [-0.00522854, -0.35613713, -0.93441908]])


    def position_from_cam_to_base(self,pos_cam):
        return self.R_cb@pos_cam + self.T_cb



    def  solvePnPAruco(self,corners):
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
            [self.cube_size, 0, 0],
            [self.cube_size, self.cube_size, 0],
            [0, self.cube_size, 0]
        ], dtype=np.float32)

        success, rvec, tvec = cv.solvePnP(obj_points, img_points, self.camera_matrix, self.dist_coeffs)
        R, _ = cv.Rodrigues(rvec)
        center_obj= np.array([[self.cube_size/2, self.cube_size/2, self.cube_size/2]], dtype=np.float32)
        center_cam = R @ center_obj.T + tvec
        center_cam = center_cam.flatten()

        if not success:
            return None

        return [ center_cam , rvec , tvec ]

    # déterminer la position de l'objet dans le repère de la base
    def get_object_position(self, frame):

        # paramètres de détection
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(self.arucodict, parameters)

        # Converting to grayscale
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

        # détection des marqueurs
        corners, ids, rejected = detector.detectMarkers(gray)
        print(ids)
        if ids is not None:
            # dessiner les marqueurs détectés
            aruco.drawDetectedMarkers(frame, corners, ids)

            solved = self.solvePnPAruco(corners)
            center_cam = solved[0]
            print(center_cam)
            center_base = self.position_from_cam_to_base(center_cam)
            print(center_base)
            
            cv.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, solved[1], solved[2], 3)
        else:
            center_base = None
        return center_base

        
