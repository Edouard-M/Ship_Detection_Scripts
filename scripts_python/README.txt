----- Projet SHIPMARK de fin d'études 2023 -----

Ce dossier contient 4 algorithmes servant d'outils pour les annotations du modèle :
  -  1 algorithme d'annotation automatique 
  -  1 algorithme de mesure des résultats permettant d'avoir les statistiques de réussite du modèle
  -  2 algorithmes de conversion, permettant de convertir des annotations YOLO-V7 en COCO ou bien COCO en YOLO-V7

Dans ces dossiers 2 modèles d'annotation sont présents :

YOLO-V7 :
-- Exemple du format YOLO-V7 : (1 fichier par image satellite, le nom du fichier au format [num_image].txt)
0 [bateau 1 - X avant] [bateau 1 - Y avant]
3 [bateau 1 - X arrière Gauche] [bateau 1 - Y arrière Gauche]
2 [bateau 1 - X arrière Droit] [bateau 1 - Y arrière Droit]
0 [bateau 2 - X avant] [bateau 2 - Y avant]
3 [bateau 2 - X arrière Gauche] [bateau 2 - Y arrière Gauche]
2 [bateau 2 - X arrière Droit] [bateau 2 - Y arrière Droit]

COCO :
1 fichier "dataset.json" comprenant les annotations de plusieurs images