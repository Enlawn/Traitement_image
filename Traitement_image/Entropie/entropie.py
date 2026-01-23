import cv2
import numpy as np

# Fonction pour calculer l'entropie d'une image
def calcul_entropie(image):
    """
    Calcule l'entropie d'une image en niveaux de gris
    """

    # Calcul de l'histogramme (256 niveaux de gris)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Normalisation de l'histogramme pour obtenir des probabilités
    hist = hist / hist.sum()

    # Suppression des valeurs nulles pour éviter log(0)
    hist = hist[hist > 0]

    # Calcul de l'entropie
    entropie = -np.sum(hist * np.log2(hist))

    return entropie


# Chargement des deux images en niveaux de gris
image1 = cv2.imread("coeur.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("coeur.jpg", cv2.IMREAD_GRAYSCALE)

# Calcul de l'entropie pour chaque image
entropie1 = calcul_entropie(image1)
entropie2 = calcul_entropie(image2)

# Affichage des résultats
print("Entropie de l'image 1 :", entropie1)
print("Entropie de l'image 2 :", entropie2)

# Comparaison des entropies
if entropie1 > entropie2:
    print("L'image 1 contient plus d'information (plus de détails).")
elif entropie1 < entropie2:
    print("L'image 2 contient plus d'information (plus de détails).")
else:
    print("Les deux images ont la même entropie.")
