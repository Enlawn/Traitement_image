import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



image=cv.imread("coeur.jpg")
cv.imshow('image originale',image)



#=====================
# Transformation géométrique / morphologique: dilation, érosion, changement d'échelle, rotation
#lien vers la doc : https://docs.opencv.org/4.x/db/df6/tutorial_erosion_dilatation.html

# erosion
# objectif : récupèrer le code qui permet de modifier le parametrage pour l'appliquer à la modification de la luminosité et du contraste de l'image | sera fait en projet perso, pas le temps de le faire dans le cadre du cours
source = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# correction réalisée ici : copie de l'image source pour ne pas modifier l'originale à chaque itération de la trackbar
im1 = source.copy()
im2 = source.copy()
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
	Image_erodee = cv.erode(im1, element)
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
	Image_dilate = cv.dilate(im2, element)
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

taille_morph = 15  # taille initiale de l'élément structurant, modifié par la trackbar
title_trackbar_element_shape = 'Element:\n 1: Rect \n 2: Cross \n 3: Ellipse \n 4: Diamond'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
title_grad_window = 'Gradient morphologique'

def gradient_morph(val):
	taille_morph = cv.getTrackbarPos(title_trackbar_kernel_size, title_grad_window)
	grad_shape = element_structurant(cv.getTrackbarPos(title_trackbar_element_shape, title_grad_window))
	element_grad = cv.getStructuringElement(grad_shape, (2*taille_morph+1, 2*taille_morph+1), (taille_morph, taille_morph))
	Gradient_morphologique = cv.morphologyEx(source, cv.MORPH_GRADIENT, element_grad)
	cv.imshow(title_grad_window, Gradient_morphologique)

# création de la fenêtre et des trackbars
cv.namedWindow(title_grad_window)
cv.createTrackbar(title_trackbar_element_shape, title_grad_window, 1, max_element_structurant, gradient_morph)
cv.createTrackbar(title_trackbar_kernel_size, title_grad_window, 0, taille_noyau_max, gradient_morph)
# appel initial pour afficher l'image
gradient_morph(0) 



# ======================
# attente avant fermeture des fenetres
cv.waitKey(0)
cv.destroyAllWindows()
