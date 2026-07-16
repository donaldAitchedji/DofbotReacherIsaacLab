import cv2 as cv
import numpy as np



def determinSeuil(gray):
    #moy=0
    min=255
    max=0   
    #taille=gray.shape[0]*gray.shape[1]
    for j in range(gray.shape[0]):
       for i in  range(gray.shape[1]):
            if gray[j,i] >max:
                max=gray[j,i]
            if min>gray[j,i]:
                min=gray[j,i]
            #moy+=gray[j,i]
    mediane=(min+max)//2
    return mediane



def seuillage(gray,seuil_gris):
    img=gray.copy()
    for j in range(gray.shape[0]):
       for i in  range(gray.shape[1]):
            if gray[j,i] >=seuil_gris:
                img[j,i]=255
            else:
               img[j,i]=0
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

    """
    maskColor = cv.bitwise_or(maskColor1, maskColor2)
    
    result = cv.bitwise_and(frame, frame, mask = maskColor)

    """

    cv.imshow('maskColor1', maskColor1)
    cv.imshow('maskColor2', maskColor2)

    cv.imshow('maskColor', maskColor)
    
    cv.imshow('Result',result)
    
    return result



i=0
while i!=16:
    if i==7:
        i+=1
        continue
    if i>9:
        img=cv.imread(f'Photos/imagesProjetLicence/imageSegmentationCouleur_00{i}.jpg')
    else:
        img=cv.imread(f'Photos/imagesProjetLicence/imageSegmentationCouleur_000{i}.jpg')
    cv.imshow('Image',img)


    results=segmentationColor(img)
    
    
    # Converting to grayscale
    gray=cv.cvtColor(results,cv.COLOR_BGR2GRAY)
    cv.imshow('Gray',gray)

    #appliquer un flou gaussien pour rendre l'image moins floue
    blur7=cv.GaussianBlur(gray,(7,7),cv.BORDER_DEFAULT)
    cv.imshow('Blur',blur7)


    #Appliquer un seuilage pour que les contours de l'objet soient plus facilement discernables par le canny
    seuil_gris=determinSeuil(blur7)
    print("Seuil",seuil_gris)
    blur_seuille=seuillage(blur7,seuil_gris)
    cv.imshow('Blur_seuille',blur_seuille)
    #Traitement de l'image
    kernel=np.ones((3, 3), np.uint8)
    blur_seuille=cv.dilate(blur_seuille, kernel, iterations=4)  
    blur_seuille=cv.erode(blur_seuille, kernel, iterations=4)  
    cv.imshow('Blur_seuille_d_e',blur_seuille)

    #Edge cascade
    canny=cv.Canny(blur_seuille,50,125)
    cv.imshow('Canny Edges',canny)

    #Determination des contours
    contours, hierarchy = cv.findContours(
        canny,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    #Determination du contour principal
    main_contour = max(contours, key=cv.contourArea)
    rect=cv.minAreaRect(main_contour)
    print(rect)
    (cx,cy),_,_=rect
    cx=int(cx)
    cy=int(cy)
    box = cv.boxPoints(rect)   # 4 sommets
    box = box.astype(int)
    print(box)



    cv.circle(img, (cx,cy),2, (255,0,0),-1) #BLUE
    cv.drawMarker(img, box[0], (255,0,0), cv.MARKER_CROSS, 30, 1) #BLUE
    cv.drawMarker(img, box[1], (0,0,255), cv.MARKER_CROSS, 30, 1) #RED
    cv.drawMarker(img, box[2], (255,255,255), cv.MARKER_CROSS, 30, 1) #white
    cv.drawMarker(img, box[3], (0,0,0), cv.MARKER_CROSS, 30, 1) #black
    cv.drawContours(img, [box], 0, (0, 255, 0), 2)
    cv.imshow('Rect Area',img)

    cv.waitKey(0)
    i+=1
