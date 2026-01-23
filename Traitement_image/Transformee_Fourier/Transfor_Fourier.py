import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt




image=cv.imread("coeur.jpg")
cv.imshow('image originale',image)




#=====================
# Transformée de Fourier discrète (TFD) 
# conversion de l'image en niveau de gris 
imf = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# vérification des caractéristiques de l'image en niveaux de gris
print(imf.dtype)
print(imf.shape)

# TF = décomposition en fonction sinusoïdales (sinus et cosinus) de l'image en entrée
# j'apprends un truc : pour certaines valeurs de tailles d'images, la TFD est plus rapide, du coup on va redimensionner l'image pour que ses dimensions soient des puissances de 2
# optimisation de la taille de l'image pour la TFD / partie non obligatoire
rows, cols = imf.shape
new_rows = cv.getOptimalDFTSize(rows)
new_cols = cv.getOptimalDFTSize(cols)
# on ajoute des bordures noires (remplissage par des zéros)
bordure = cv.copyMakeBorder(imf, 0, new_rows - rows, 0, new_cols - cols, cv.BORDER_CONSTANT, value=[0, 0, 0])
# https://docs.opencv.org/4.x/d8/d01/tutorial_discrete_fourier_transform.html
# transformée de fourier = nombre complexe en sortie, chaque pixel posséde donc 2 valeurs (partie réelle et partie imaginaire) -> on doit créer une image avec 2 canaux
partR_partIm_mono_canal = [np.float32(bordure) , np.float32(np.zeros(bordure.shape))]

Image_complexe = cv.merge(partR_partIm_mono_canal)  # fusion des 2 canaux en une seule image a 2 canaux avec la fonction cv.merge

# vérification des caractéristiques de l'image complexe parce que j'avais une erreur = error: (-215:Assertion failed) type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 in function 'cv::dft'
# correction : j'utilisais l'image d'origine et non pas l'image en niveaux de gris dans mon padding, le type de paramètre d'entrée de la fonction cv.dft n'étais pas respecté (j'avais 6 canaux au lieu de 2)

print(Image_complexe.dtype) #float32 
print(Image_complexe.shape) #(hauteur, largeur, nb_canaux) pour que la TFD fonctionne, nb_canaux doit être égal à 2

# calcul de la TFD
# calcul de la TFD en place avec la fonction cv.dft(source, destination = source) -> le résultat est stocké dans la matrice d'entrée Image_complexe = gain de mémoire
# pour un soucis de clarté (du moins pour moi), je préfère stocker le résultat dans une nouvelle variable contrairement à ce qui est fait dans la doc
image_TF = cv.dft(Image_complexe)  
# séparation des 2 canaux (partie réelle et partie imaginaire)
partieR, partieI = cv.split(image_TF)
# calcul du spectre de magnitude
magnitude_spectre = cv.magnitude(partieR, partieI)
# passage en échelle logarithmique pour une meilleure visualisation
magnitude_spectre += 1  # on ajoute 1 pour éviter log(0)
# une autre façon de faire est de créer une matrice de 1 et de l'additionner au spectre de magnitude
# -> passage en logarithme
magnitude_spectre = cv.log(magnitude_spectre)
# affichage du spectre de magnitude : on doit éliminer les bords noirs ajoutés
magnitude_spectre = magnitude_spectre[0:rows, 0:cols]
cv.normalize(magnitude_spectre, magnitude_spectre, 0, 1, cv.NORM_MINMAX)  # normalisation pour l'affichage
cv.imshow('Spectre de magnitude', magnitude_spectre)


# Pour la Transformée de Fourier inverse (TFI), on repart de l'image complexe obtenue après la TFD
# doc de la fonction cv.dtf
# https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gadd6cf9baf2b8b704a11b5f04aaf4f39d
# doc pour le paramètre flags de la fonction cv.dft
# https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gaf4dde112b483b38175621befedda1f1c

# deux ecriture équivalentes : cv.idtf(image_TF) ou bien cv.dft(image_TF, flags=cv.DFT_INVERSE | cv.DFT_SCALE)

# à modifier pour reconstruire complétement l'image d'origine
image_TFI =cv.dft(image_TF, flags=cv.DFT_INVERSE | cv.DFT_SCALE)  # le résultat est stocké dans Image_TFI
image_TFI_partR, image_TFI_partI = cv.split(image_TFI)  # séparation des 2 canaux
# l'image d'origine est réelles, on ne garde que la partie réelle
image_reconstruite = image_TFI_partR
# on élimine les bordures noires ajoutées
image_reconstruite = image_reconstruite[0:rows, 0:cols]

# normalisation pour l'affichage
# cv.normalize(image_reconstruite, image_reconstruite, 0, 255, cv.NORM_MINMAX) # image reconstruite en nuance de gris
# cv.imshow('Image reconstruite après TFI', image_reconstruite)

# conversion pour affichage (SANS normalisation)
image_reconstruite_aff = np.uint8(np.clip(image_reconstruite, 0, 255))

cv.imshow('Image reconstruite après TFI', image_reconstruite_aff)




# ======================
# attente avant fermeture des fenetres
cv.waitKey(0)
cv.destroyAllWindows()
