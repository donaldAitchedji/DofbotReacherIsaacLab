import cv2
import cv2.aruco as aruco

# création un dictionnaire DICT_4X4_100 déjà prédéfinie 
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100) 

# génération de marqueur d'ID 23 et de 113X113 pixels 
marker = aruco.generateImageMarker(aruco_dict, 23, 113)

# enregistrement du marqueur 
cv2.imwrite("marker.png", marker)
