----- algorithme d'annotation automatique -----

Cet algorithme annote automatiquement les images données en input, 
il se base sur la traînée des navires et ne sélectionne que les images sur lesquelles l'algorithme est le plus confiant
Cet algorithme prend en input les résultats du premier modèle de détection du kaggle permettant de trouver les boîtes englobantes des bateaux

 - Le dossier "input" comprend :
     - Le dossier "dataset" : contient toutes les images du dataset
     - Le dossier "segmentation_csv" : contient le document .csv contenant la liste des images annotées par l'algorithme du kaggle du "ship detection challenge" 
     - Le dossier "run" : contient les images d'input destinées à être annotées automatiquement

 - Le dossier "output" comprend :
     - Le dossier "annotations" : contient les fichiers de résultat de l'annotation au format YOLO-V7 (1 fichier par image)
     - Le dossier "images_VALID" : contient les images de DEBUG des images annotées de façon concluante
     - Le dossier "images_NOT_VALID" : contient les images de DEBUG des images annotées de façon non concluante (l'algorithme ne s'est pas jugé assez confiant sur ces images)




-- Exemple du format YOLO-V7 : (1 fichier par image satellite, le nom du fichier au format [num_image].txt)
0 [bateau 1 - X avant] [bateau 1 - Y avant]
3 [bateau 1 - X arrière Gauche] [bateau 1 - Y arrière Gauche]
2 [bateau 1 - X arrière Droit] [bateau 1 - Y arrière Droit]
0 [bateau 2 - X avant] [bateau 2 - Y avant]
3 [bateau 2 - X arrière Gauche] [bateau 2 - Y arrière Gauche]
2 [bateau 2 - X arrière Droit] [bateau 2 - Y arrière Droit]