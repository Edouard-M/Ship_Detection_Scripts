----- Programme de mesure de résultats -----

Ce programme permet d'afficher dans la console les statistiques du modèle mesuré.
 - le dossier "input" comprend les annotations ou prédictions à mesurer, le format d'entrée est YOLO-V7, 
   (il est possible de convertir une annotation COCO en YOLO-V7 via les programmes de conversion) (1 fichier par image annotée)
 - le dossier "verite_terrain" comprend les annotations manuelles du dataset permettant de valider les résultats.
   (Ce dossier sert de référence pour les mesures) (1 fichier par image annotée)


Ce programme mesure et affiche ainsi dans la console :
 - la précision en pixels de keypoints
 - l'écart-type de ces points
 - la robustesse du modèle (pourcentage de bateaux trouvés dans le bon sens parmi les bateaux trouvés)
 - le pourcentage de bateaux trouvés par rapport à la totalité des bateaux



-- Exemple du format YOLO-V7 : -- -- (1 fichier par image satellite, le nom du fichier au format [num_image].txt)
0 [bateau 1 - X avant] [bateau 1 - Y avant]
3 [bateau 1 - X arrière Gauche] [bateau 1 - Y arrière Gauche]
2 [bateau 1 - X arrière Droit] [bateau 1 - Y arrière Droit]
0 [bateau 2 - X avant] [bateau 2 - Y avant]
3 [bateau 2 - X arrière Gauche] [bateau 2 - Y arrière Gauche]
2 [bateau 2 - X arrière Droit] [bateau 2 - Y arrière Droit]