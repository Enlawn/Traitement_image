import numpy as np
import cv2 as cv

image=cv.imread("coeur.jpg")
cv.imshow("image originale", image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
src = gray.copy()
prewittx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], np.float32)  

prewitty = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]], np.float32)  

# application des masques de Prewitt
# comment s'assurer qu'on a bien prewittx qui s'applique sur les variations selon x et prewitty sur y ?
# la direction du masque définiee la direction de détection des contours
# pour prewittx : les coefficients varient de gauche à droite, donc selon l'axe horizontale( selon x donc)
# pour prewitty : les coefficients varient de haut en bas, donc selon l'axe vertical (selon y donc)

#la fonction cv.filter2D permet de réaliser le produit de convolution entre l'image et le masque
masqueselon_x = cv.filter2D(src, cv.CV_32F, prewittx)
masqueselon_y = cv.filter2D(src, cv.CV_32F, prewitty)

# ne voyant pas une réelle différence entre sobel et prewitt, j'affiche les images des gradients selon x et y 
# mais meme comme ça, on ne voit pas de différence flagrante entre les 2 méthodes !!
cv.imshow("image gradient selon x prewitt", masqueselon_x)
cv.imshow("image gradient selon y prewitt", masqueselon_y)


# une fois le gradient calculé selon x et y, on peut combiner les 2 pour obtenir le gradient total, l'amplitude du gradient nous permet de détecter les contours
amplitude = cv.magnitude(masqueselon_x, masqueselon_y)

# les calculs mathématiques se font toujours en float32 pour éviter les débordements, mais pour l'affichage on doit convertir en uint8
dst1 = cv.convertScaleAbs(amplitude)

cv.imwrite("image_contour_prewitt.jpg", dst1)
cv.imshow("image contour prewitt", dst1)


#=====================
# Filtre de Sobel
# on utilise le même raisonnement que pour Prewitt

src2 = gray.copy()
sobelx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], np.float32)  

sobely = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], np.float32)  

# application des masques de Sobel
masqueselon_x_sobel = cv.filter2D(src2, cv.CV_32F, sobelx)
masqueselon_y_sobel = cv.filter2D(src2, cv.CV_32F, sobely)

cv.imshow("image gradient selon x sobel", masqueselon_x_sobel)
cv.imshow("image gradient selon y sobel", masqueselon_y_sobel)

amplitude_sobel = cv.magnitude(masqueselon_x_sobel, masqueselon_y_sobel)
dst2 = cv.convertScaleAbs(amplitude_sobel)
cv.imwrite("image_contour_sobel.jpg", dst2)
cv.imshow("image contour sobel", dst2)






#=====================
cv.waitKey(0)
cv.destroyAllWindows()
