----- Conversion d'annotations format YOLO-V7 vers format COCO -----
Cet algorithme permet de convertir des annotations format YOLO-V7 vers un seul fichier format COCO.
Le format COCO est utilisé pour entraîner notre modèle de deep learning fonctionnant sous ce format
Le format YOLO-V7 est utilisé pour mesurer les résultats du modèle ainsi qu'en sortie de l'annotation automatique

- Le dossier "input_YOLO" comprend alors les fichiers d'annotations destinés à être transformés, tous les fichiers de ce dossier seront alors automatiquement convertis
- Le dossier "output_COCO" comprend le fichier "dataset.json" comprenant l'annotation de l'ensemble des fichiers convertis ainsi que les images


-- Exemple du format YOLO-V7 : -- (1 fichier par image satellite, le nom du fichier au format [num_image].txt)
0 [bateau 1 - X avant] [bateau 1 - Y avant]
3 [bateau 1 - X arrière Gauche] [bateau 1 - Y arrière Gauche]
2 [bateau 1 - X arrière Droit] [bateau 1 - Y arrière Droit]
0 [bateau 2 - X avant] [bateau 2 - Y avant]
3 [bateau 2 - X arrière Gauche] [bateau 2 - Y arrière Gauche]
2 [bateau 2 - X arrière Droit] [bateau 2 - Y arrière Droit]