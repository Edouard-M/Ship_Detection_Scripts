----- Conversion d'annotations format COCO vers format YOLO-V7 -----
Cet algorithme permet de convertir des annotations format COCO vers un seul fichier format YOLO-V7.
Le format COCO est utilisé pour entraîner notre modèle de deep learning fonctionnant sous ce format
Le format YOLO-V7 est utilisé pour mesurer les résultats du modèle ainsi qu'en sortie de l'annotation automatique

- Le dossier "input_COCO" comprend alors le fichier "dataset.json" destiné à être converti
- Le dossier "output_YOLO" comprend l'ensemble des fichiers convertis au format YOLO-V7, un fichier par image satellite


-- Exemple du format YOLO-V7 : -- (1 fichier par image satellite, le nom du fichier au format [num_image].txt)
0 [bateau 1 - X avant] [bateau 1 - Y avant]
3 [bateau 1 - X arrière Gauche] [bateau 1 - Y arrière Gauche]
2 [bateau 1 - X arrière Droit] [bateau 1 - Y arrière Droit]
0 [bateau 2 - X avant] [bateau 2 - Y avant]
3 [bateau 2 - X arrière Gauche] [bateau 2 - Y arrière Gauche]
2 [bateau 2 - X arrière Droit] [bateau 2 - Y arrière Droit]