import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt




# ouverture et affichage de l'image
image=cv.imread("coeur.jpg")
cv.imshow('image originale',image)

# ===================== 
# lien vers la documentation : https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html
# on modifie le contraste et la luminosite de l'image
# il existe 3 façon de le faire  
# 1. en utilisant la fonction cv.convertScaleAbs
# 2. en utilisant une fonction linéaire : image_corrigée = alpha * image + beta qui parcourt chaque pixel de l'image
# 3. en utilisant une fonction non linéaire : la correction gamma

# j'avais une erreur ici : je n'utilisais pas le .copy() ce qui modifiait l'image d'origine et faussait toutes mes autres transformations
# à vérifier dans toutes les autres parties du code !!
im1 = image.copy()

alpha =float (input('entrer la valeur de alpha pour modifier le contraste : '))  #pour le contraste
beta = float (input('entrer la valeur de beta pour modifier la luminosite : '))  #pour la luminosite 

for i in range (im1.shape[0]):
	for j in range (im1.shape[1]):
		for c in range (im1.shape[2]):
			im1[i,j,c] = np.clip(alpha * im1[i,j,c] + beta, 0, 255)

cv.imshow('contraste et luminosite modifie', im1)

#=====================
# correction gamma

im6 =  image.copy()
gamma = float (input('entrer la valeur de gamma pour la correction gamma(luminosité) : '))

lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

res = cv.LUT(im6, lookUpTable)
# on utilise la fonction cv.LUT (Look Up Table) pour appliquer la correction gamma à l'image
# concretement, la lookuptable est une table de correspondance qui stocke tout les valeurs possible de pixel (0-255) et leur valeur corrigée en fonction de la formule de correction gamma
# on applique ensuite cette table à chaque pixel de l'image d'origine pour obtenir l'image corrigée, ce qui est plus efficace que de recalculer la formule pour chaque pixel individuellement
# de plus, cette méthode permet d'éviter les erreurs de typage et de dépassement de valeur en s'assurant que toutes les valeurs restent dans la plage valide (0-255) grâce à l'utilisation de np.clip lors de la création de la lookuptable
# et enfin, l'utilisation de cv.LUT permet d'appliquer la transformation sur l'ensemble des canaux de couleurs de l'image sans devoir traiter chaque canal séparément
# = gain de temps, on évite les erreurs de type (que je fais beaucoup trop souvent) et les oublie de couche de couleur non modifiée !!
cv.imshow('correction gamma(luminosite)', res)


#=====================
# on change la colormap de l'image : lien vers la doc https://docs.opencv.org/4.13.0/d3/d50/group__imgproc__colormap.html

# 1. conversion en niveaux de gris
im2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('niveaux de gris', im2)
# 2. conversion en fausses couleurs : colorimetrie Parula 
im3 = cv.applyColorMap(image, cv.COLORMAP_PARULA)
cv.imshow('fausses couleurs', im3)

#===================== sans tutoriel, application de ce qu'on a vu en cours d'image et vision 
# seuillage de l'image -> on passe de l'image en niveaux de gris a une image binaire
new_image = im2.copy()   

seuil= 50

for i in range (im2.shape[0]):
	for j in range(im2.shape[1]):
		if (new_image[i,j] < seuil):
			new_image[i,j] = 0
		else:
			new_image[i,j] = 255
cv.imshow('Seuillage', new_image)


#=====================
# inversion des couleurs de l'image
im4 = cv.bitwise_not(image)
cv.imshow('inversion des couleurs', im4)






























# ======================
# attente avant fermeture des fenetres
cv.waitKey(0)
cv.destroyAllWindows()
