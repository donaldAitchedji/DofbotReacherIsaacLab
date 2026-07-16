import numpy as np
import cv2 as cv

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


# Taille réelle des coté du cube en mètres
cube_edge_size = 0.03

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