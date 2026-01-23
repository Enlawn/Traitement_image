import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



image=cv.imread("coeur.jpg")
cv.imshow('image originale',image)
#=====================
source = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


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



# Détection de contours
def detect_contours(val):
	image_canny = CannyThreshold(250)  # on utilise un seuil fixe pour la détection de contours
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


# Histogramme de l'image en utilisant Matplotlib
# en niveau de gris
plt.figure()
plt.hist((cv.cvtColor(image, cv.COLOR_BGR2GRAY)).ravel(),256,[0,256]) 
plt.title('Histogramme en niveaux de gris de l\'image')
plt.xlabel('Intensité des pixels')
plt.ylabel('Nombre de pixels')

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
plt.title('Histogramme de l\'image apres egalisation d\'histogramme avec CLAHE')
plt.xlabel('Intensité des pixels')
plt.ylabel('Nombre de pixels')

plt.show()


# ======================
# attente avant fermeture des fenetres
cv.waitKey(0)
cv.destroyAllWindows()