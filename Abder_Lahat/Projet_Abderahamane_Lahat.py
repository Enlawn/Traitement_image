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


im1 = image

alpha =float (input('entrer la valeur de alpha pour modifier le contraste : '))  #pour le contraste
beta = float (input('entrer la valeur de beta pour modifier la luminosite : '))  #pour la luminosite 

for i in range (im1.shape[0]):
	for j in range (im1.shape[1]):
		for c in range (im1.shape[2]):
			im1[i,j,c] = np.clip(alpha * im1[i,j,c] + beta, 0, 255)

cv.imshow('contraste et luminosité modifiée', im1)

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
new_image = im2   

seuil= 50

for i in range (im2.shape[0]):
	for j in range(im2.shape[1]):
		if (new_image[i,j] < seuil):
			new_image[i,j] = 0
		else:
			new_image[i,j] = 255
cv.imshow('new_image', new_image)


#=====================
# inversion des couleurs de l'image
im4 = cv.bitwise_not(image)
cv.imshow('inversion des couleurs', im4)






#=====================
# Transformation géométrique / morphologique: dilation, érosion, changement d'échelle, rotation
#lien vers la doc : https://docs.opencv.org/4.x/db/df6/tutorial_erosion_dilatation.html

# erosion
# objectif : récupèrer le code qui permet de modifier le parametrage pour l'appliquer à la modification de la luminosité et du contraste de l'image
source = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
taille_erosion = 0  # taille initiale de l'érosion, modifié par la trackbar
# Pourquoi un nombre impair ? Parce que l'élément structurant doit avoir un centre bien défini et qu'avec une taille paire, il n'y a pas de pixel central.

# 'Element_structurant:\n 1: Rect \n 2: Cross \n 3: Ellipse \n 4: Diamond'
max_element_structurant = 4 # ne pas mettre 0 ici sinon la trackbar ne fonctionne pas
def element_structurant(val):
	if val==1:
		return cv.MORPH_RECT
	elif val==2:
		return cv.MORPH_CROSS
	elif val==3:
		return cv.MORPH_ELLIPSE
	else:
		return cv.MORPH_DIAMOND #comment ça y'a pas de morph_diamond dans opencv IDE de mes 2 ?!
	
taille_noyau_max = 23

title_trackbar_element_shape = 'Element:\n 1: Rect \n 2: Cross \n 3: Ellipse \n 4: Diamond'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'

title_erosion_window = 'Erosion'

def erosion(val):
	taille_erosion = cv.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window)
	erosion_shape = element_structurant(cv.getTrackbarPos(title_trackbar_element_shape, title_erosion_window))

	element = cv.getStructuringElement(erosion_shape,(2*taille_erosion+1, 2*taille_erosion+1),(taille_erosion, taille_erosion))
	Image_erodee = cv.erode(source, element)
	cv.imshow(title_erosion_window, Image_erodee)

# création de la fenêtre et des trackbars
cv.namedWindow(title_erosion_window)
cv.createTrackbar(title_trackbar_element_shape, title_erosion_window, 1, max_element_structurant, erosion)
cv.createTrackbar(title_trackbar_kernel_size, title_erosion_window, 0, taille_noyau_max, erosion)
# appel initial pour afficher l'image
erosion(0) 

# une autre façon de faire l'érosion sans trackbar
# element = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
# Image_erodee = cv.erode(source, element)
# cv.imshow('Image érodée sans trackbar', Image_erodee)

# dilatation

title_dilatation_window = 'Dilatation'
dilatation_size = 0  # taille initiale de la dilatation, modifié par la trackbar

def dilatation(val):
	dilatation_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_dilatation_window)
	dilatation_shape = element_structurant(cv.getTrackbarPos(title_trackbar_element_shape, title_dilatation_window))
	element = cv.getStructuringElement(dilatation_shape,(2*dilatation_size+1, 2*dilatation_size+1),(dilatation_size, dilatation_size))
	Image_dilate = cv.dilate(source, element)
	cv.imshow(title_dilatation_window, Image_dilate)

# création de la fenêtre et des trackbars
cv.namedWindow(title_dilatation_window)
cv.createTrackbar(title_trackbar_element_shape, title_dilatation_window, 1, max_element_structurant, dilatation)
cv.createTrackbar(title_trackbar_kernel_size, title_dilatation_window, 0, taille_noyau_max, dilatation)
# appel initial pour afficher l'image
dilatation(0) 



# =====================
# # une utilisation alternative de la transformation morphologique : le gradient morphologique qui est la différence entre la dilatation et l'érosion d'une image
# # lien vers la doc : https://docs.opencv.org/4.x/d3/dbe/tutorial_opening_closing_hats.html
# # une premeière approche de la détection de contours

# taille_morph = 0  # taille initiale de l'élément structurant, modifié par la trackbar
# title_trackbar_element_shape = 'Element:\n 1: Rect \n 2: Cross \n 3: Ellipse \n 4: Diamond'
# title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
# title_grad_window = 'Gradient morphologique'

# def gradient_morph(val):
# 	taille_morph = cv.getTrackbarPos(title_trackbar_kernel_size, title_grad_window)
# 	grad_shape = element_structurant(cv.getTrackbarPos(title_trackbar_element_shape, title_grad_window))
# 	element_grad = cv.getStructuringElement(grad_shape, (2*taille_morph+1, 2*taille_morph+1), (taille_morph, taille_morph))
# 	Gradient_morphologique = cv.MORPH_GRADIENT(source, element_grad)
# 	cv.imshow(title_grad_window, Gradient_morphologique)

# # création de la fenêtre et des trackbars
# cv.namedWindow(title_grad_window)
# cv.createTrackbar(title_trackbar_element_shape, title_grad_window, 1, max_element_structurant, gradient_morph)
# cv.createTrackbar(title_trackbar_kernel_size, title_grad_window, 0, taille_noyau_max, gradient_morph)
# # appel initial pour afficher l'image
# gradient_morph(0) 

# =====================

# Detection de contour

# lien vers la doc : https://docs.opencv.org/4.x/df/d0d/tutorial_find_contours.html

# lien vers la doc : https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html

max_lowThreshold = 150
window_canny_name = 'Image des contours - Canny Edge Detection'
title_canny_trackbar = 'Min Threshold:'
ratio = 3
kernel_canny_size = 3

print(source.dtype)
print(source.shape)

def CannyThreshold(val):
    low_threshold = val

	#img_blur = cv.blur(source, (3,3)) # flou basique
    img_blur = cv.GaussianBlur(source, (3,3), 1.4) # flou gaussien, valeur de sigma choisi avec la page wikipedia : https://en.wikipedia.org/wiki/Canny_edge_detector?oldid=498925521#Noise_reduction

    print(img_blur.dtype)
    print(img_blur.shape)
	# application de l'algorithme de Canny pour la détection de contours qui va comparer les gradients de l'image floutée avec les seuils
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_canny_size)
	# masque = seuillage binaire pour ne garder que les contours détectés
    mask = detected_edges != 0
    image_contour = cv.bitwise_and(source, source, mask=detected_edges) # demandé à chat GPT parce que je ne comprenais pas comment faire sachant que j'avais des typage correcte
    cv.imshow(window_canny_name, image_contour)
    return detected_edges

	

cv.namedWindow(window_canny_name)
cv.createTrackbar(title_canny_trackbar, window_canny_name , 0, max_lowThreshold, CannyThreshold)
# appel initial pour afficher l'image
CannyThreshold(0)



# détection de contours
def detect_contours(val):
	image_canny = CannyThreshold(150)  # on utilise un seuil fixe pour la détection de contours
	contours, hierarchy = cv.findContours(image_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	# création d'une image noire pour dessiner les contours
	image_contours = np.zeros_like(image)
	# dessin des contours en rouge
	for i in range(len(contours)):
		cv.drawContours(image_contours, contours, i, (0, 0, 255), 2, cv.LINE_8, hierarchy, 0)

	cv.imshow('Contours', image_contours)

cv.namedWindow('Contours')
detect_contours(0)





# =====================
image=cv.imread("coeur.jpg")


# Histogramme de l'image en utilisant OpenCV -> obligatoire pour une image en couleur
# doc : https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
# Couleurs 
plt.figure()
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
	histogramme = cv.calcHist([image],[i], None, [256], [0, 256])
	plt.plot(histogramme, color=col)
	plt.xlim([0, 256])
plt.title('Histogramme des couleurs de l\'image')
plt.xlabel('Intensité des pixels')
plt.ylabel('Nombre de pixels')
plt.show()

# Histogramme de l'image en utilisant Matplotlib
# en niveau de gris
plt.figure()
plt.hist((cv.cvtColor(image, cv.COLOR_BGR2GRAY)).ravel(),256,[0,256]) 
plt.show()

# histogramme egalisation
# doc : https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
# doc : https://en.wikipedia.org/wiki/Histogram_equalization

# directement faire une egalisation d'histogramme avec limites adaptatives avec la methode CLAHE
clahe = cv.createCLAHE()
cl1 = clahe.apply(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
cv.imwrite("Image_apres_egalisation_histogramme_CLAHE.jpg", cl1)
cv.imshow('Image après egalisation d\'histogramme avec CLAHE', cl1)

imclahe = cv.imread("Image_apres_egalisation_histogramme_CLAHE.jpg")
plt.figure()
plt.hist(imclahe.ravel(),256,[0,256]) 
plt.show()




















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
magnitude_spectre += 1  # on ajoute 1 pour éviter les problèmes de log(0)
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
cv.normalize(image_reconstruite, image_reconstruite, 0, 255, cv.NORM_MINMAX) # image reconstruite en nuance de gris
cv.imshow('Image reconstruite après TFI', image_reconstruite)






# ======================
# attente avant fermeture des fenetres
cv.waitKey(0)