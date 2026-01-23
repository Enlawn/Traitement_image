import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

####Image bruitee avec le filtre gaussien
image=cv.imread("coeur.jpg")
cv.imshow('image originale',image)
src = image.copy()
src = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # type = uint8
image1 = src
mean = 0

sigma = 25
# On en a chier pendant 1h avec s'te fonction parce que le typage de np.random n'était pas compatible avec l'affichage de l'image et on se retrouvait avec une image trop saturée/ trop lumineuse / blanche
# solution : forcer le typage de la matrice de valeurs random en uint8 !
gauss = np.random.normal(mean,sigma,image1.shape).astype(np.uint8)# type =  uint8
cv.imshow('image bruitee ',gauss)


gauss1 =  cv.add(image1 , gauss)
cg=gauss1.copy()
# gauss1 =  gauss+ image1 
# reponse à notre question :
# https://stackoverflow.com/questions/65633716/what-is-the-different-between-adding-to-images-together-using-and-the-add
gauss1 =np.uint8(gauss1)
cv.imwrite("/media/pi/ESD-USB/Master1/Traitement_image/image/image_bruit_gaussien.jpg", gauss1)
cv.imshow('image bruitee avec un Gausien',gauss1)

#vals = len(np.unique(image))
#vals = 2 ** np.ceil(np.log2(vals))
#noisy_image2 = np.random.poisson(image * vals) / float(vals)
#cv.imshow('image bruitee avec un Poisson',noisy_image2)

dst = cv.fastNlMeansDenoisingColored(cg,10,7,21,None)
cv.imwrite("/media/pi/ESD-USB/Master1/Traitement_image/image/image_debruitee_gaussien.jpg", cg)
cv.imshow('image bruitee avec un Gausien',cg)
 


print(gauss.dtype)
print(image1.dtype)





cv.waitKey(0)
cv.destroyAllWindows()
