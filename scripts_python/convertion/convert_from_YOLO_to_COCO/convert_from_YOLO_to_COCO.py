from math import *
from struct import *
import os
import json
import shutil

class coord:
    x  = 0.0
    y  = 0.0

class Bateau:
    avant     = coord()
    arriere_g = coord()
    arriere_d = coord()

    def show(self):
        print("avant :  x = ", self.avant.x, " y = ", self.avant.y)
        print("arriere_g :  x = ", self.arriere_g.x, " y = ", self.arriere_g.y)
        print("arriere_d :  x = ", self.arriere_d.x, " y = ", self.arriere_d.y)

class BateauPair:
    Bateau_True = Bateau()
    Bateau_Test = Bateau()


#--------------------- lit tout un fichier ----------------------------#
def read_file(dir, file):
    f = open(path + "/" + dir + "/" + file + ".txt", "r")
    str_file = f.read()
    f.close()

    return str_file
#---------------------------------------------------------------------#

def deplacer_keypoint_arriere(bateau):

    new_bateau = Bateau()

    decalage_avant = 25/100   # 20
    decalage_arriere = 37/100 # 30
    decalage = 45/100         # 37

    # Decalage des 2 points arriere pour pas que ce soit dans l'eau
    d_largeur_x = abs(bateau.arriere_g.x - bateau.arriere_d.x)
    d_largeur_y = abs(bateau.arriere_g.y - bateau.arriere_d.y)
    milieu_arriere_x = abs(bateau.arriere_g.y - bateau.arriere_d.y)/2
    milieu_arriere_y = abs(bateau.arriere_g.y - bateau.arriere_d.y)/2
    d_longeur_x = abs(milieu_arriere_x - bateau.avant.x)
    d_longeur_y = abs(milieu_arriere_y - bateau.avant.y)

    if(bateau.arriere_g.x < bateau.arriere_d.x):
        new_bateau.arriere_g.x = bateau.arriere_g.x + d_largeur_x*decalage
        new_bateau.arriere_d.x = bateau.arriere_d.x - d_largeur_x*decalage
    else:
        new_bateau.arriere_g.x = bateau.arriere_g.x - d_largeur_x*decalage
        new_bateau.arriere_d.x = bateau.arriere_d.x + d_largeur_x*decalage

    if(bateau.arriere_g.y < bateau.arriere_d.y):
        new_bateau.arriere_g.y = bateau.arriere_g.y + d_largeur_y*decalage
        new_bateau.arriere_d.y = bateau.arriere_d.y - d_largeur_y*decalage
    else:
        new_bateau.arriere_g.y = bateau.arriere_g.y - d_largeur_y*decalage
        new_bateau.arriere_d.y = bateau.arriere_d.y + d_largeur_y*decalage


    # Décalage du point avant vers le centre
    if(bateau.avant.x < milieu_arriere_x):
        new_bateau.avant.x = bateau.avant.x + d_longeur_x*decalage_avant
    else:
        new_bateau.avant.x = bateau.avant.x - d_longeur_x*decalage_avant

    if(bateau.avant.y < milieu_arriere_y):
        new_bateau.avant.y = bateau.avant.y + d_longeur_y*decalage_avant
    else:
        new_bateau.avant.y = bateau.avant.y - d_longeur_y*decalage_avant


    # Décalage du point arriere vers le centre
    if(bateau.arriere_g.x < milieu_arriere_x):
        new_bateau.arriere_g.x = bateau.arriere_g.x + d_largeur_x*decalage_arriere
    else:
        new_bateau.arriere_g.x = bateau.arriere_g.x - d_largeur_x*decalage_arriere

    if(bateau.arriere_g.y < milieu_arriere_y):
        new_bateau.arriere_g.y = bateau.arriere_g.y + d_largeur_y*decalage_arriere
    else:
        new_bateau.arriere_g.y = bateau.arriere_g.y - d_largeur_y*decalage_arriere


    if(bateau.arriere_d.x < milieu_arriere_x):
        new_bateau.arriere_d.x = bateau.arriere_d.x + d_largeur_x*decalage_arriere
    else:
        new_bateau.arriere_d.x = bateau.arriere_d.x - d_largeur_x*decalage_arriere

    if(bateau.arriere_d.y < milieu_arriere_y):
        new_bateau.arriere_d.y = bateau.arriere_d.y + d_largeur_y*decalage_arriere
    else:
        new_bateau.arriere_d.y = bateau.arriere_d.y - d_largeur_y*decalage_arriere



    return new_bateau


#--------------------- lit les bateaux d'un fichier ----------------------------#
def find_bateaux(string):
    bateaux = []
    string_tab = string.split("\n")

    for j in range(int(len(string_tab)/3)):
        bateau = Bateau()
        bateau.avant     = coord()
        bateau.arriere_d = coord()
        bateau.arriere_g = coord()
        for i in range(j*3, ((j*3)+3)):
            string_values = string_tab[i].split(" ")
            if(int(string_values[0]) == 0):
                bateau.avant.x  = string_values[1]
                bateau.avant.y  = string_values[2]
            if(int(string_values[0]) == 2):
                bateau.arriere_d.x  = string_values[1]
                bateau.arriere_d.y  = string_values[2]
            if(int(string_values[0]) == 3):
                bateau.arriere_g.x  = string_values[1]
                bateau.arriere_g.y  = string_values[2]

        bateau.avant.x = int(float(bateau.avant.x)*768)
        bateau.avant.y = int(float(bateau.avant.y)*768)
        bateau.arriere_g.x = int(float(bateau.arriere_g.x)*768)
        bateau.arriere_g.y = int(float(bateau.arriere_g.y)*768)
        bateau.arriere_d.x = int(float(bateau.arriere_d.x)*768)
        bateau.arriere_d.y = int(float(bateau.arriere_d.y)*768)

        bateau = deplacer_keypoint_arriere(bateau)

        bateaux.append(bateau)

    return bateaux
#-------------------------------------------------------------------------------#

def find_bbox(bateau):

    width_bottom  = abs(bateau.arriere_g.x - bateau.arriere_d.x)
    height_bottom = abs(bateau.arriere_g.y - bateau.arriere_d.y)

    point_avant_1_x = int(bateau.avant.x - (width_bottom/2))
    if(point_avant_1_x < 1):
        point_avant_1_x = 1
    if(point_avant_1_x > 767):
        point_avant_1_x = 767

    point_avant_1_y = int(bateau.avant.y - (height_bottom/2))
    if(point_avant_1_y < 1):
        point_avant_1_y = 1
    if(point_avant_1_y > 767):
        point_avant_1_y = 767

    point_avant_2_x = int(bateau.avant.x + (width_bottom/2))
    if(point_avant_2_x < 1):
        point_avant_2_x = 1
    if(point_avant_2_x > 767):
        point_avant_2_x = 767

    point_avant_2_y = int(bateau.avant.y + (height_bottom/2))
    if(point_avant_2_y < 1):
        point_avant_2_y = 1
    if(point_avant_2_y > 767):
        point_avant_2_y = 767

    x_list = [bateau.arriere_g.x, bateau.arriere_d.x, point_avant_1_x, point_avant_2_x]
    y_list = [bateau.arriere_g.y, bateau.arriere_d.y, point_avant_1_y, point_avant_2_y]

    x_min = 768
    y_min = 768
    x_max = 0
    y_max = 0
    for i in range(len(x_list)):
        if(x_list[i] > x_max):
            x_max = x_list[i]
        if(x_list[i] < x_min):
            x_min = x_list[i]
        if(y_list[i] > y_max):
            y_max = y_list[i]
        if(y_list[i] < y_min):
            y_min = y_list[i]

    width  = int(abs(x_max - x_min))
    height = int(abs(y_max - y_min))



    #---------- Ajoute des marges à la BBOX pour pas qu'elle soit collée aux points ----------#
    marge = 4 # Marges de la bbox en pixels

    x_min  = x_min - marge
    if(x_min < 1):
        x_min = 1
    y_min  = y_min - marge
    if(y_min < 1):
        y_min = 1
    width  = width + int(marge*2)
    if((width + x_min) > 767):
        width = 767 - x_min
    height = height + int(marge*2)
    if((height + y_min) > 767):
        height = 767 - y_min
    #---------- ----------------------------------------------------------------- ----------#

    bbox = [x_min, y_min, width, height] # [xmin, ymin, width, height] 

    return bbox


def find_segmentation(bbox):

    xmin = bbox[0]
    ymin = bbox[1]
    width = bbox[2]
    height = bbox[3]
    xmax = xmin + width
    ymax = ymin + height

    segmentation = [xmin, ymax, xmin, ymin, xmax, ymin , xmax, ymax]

    return segmentation


def read_all_files(dir_from):
    print()
    image_id = 0
    id = 0
    bateaux = []
    all_files = os.listdir(path  + "/" + dir_from + "/")
    for file in all_files:
        bateaux = find_bateaux(read_file(dir_from, file[0:-4]))
        print("File = ", file)
        print("Bateaux len = ", len(bateaux))
        image_id += 1
        path_image = "/datasets/" + str(file[0:-4]) + ".jpg"
        file_name = str(file[0:-4]) + ".jpg"
        for i in range(len(bateaux)):

            bbox = find_bbox(bateaux[i])
            area = int(bbox[2] * bbox[3])
            segmentation = find_segmentation(bbox)
            id += 1

            if(image_id == 1 and id == 1):
                dictionary = {
                    "images": [
                        {
                            "id": image_id,
                            "dataset_id": 1,
                            "category_ids": [],
                            "path": path_image,
                            "width": 768,
                            "height": 768,
                            "file_name": file_name,
                            "annotated": False,
                            "annotating": [],
                            "num_annotations": 0,
                            "metadata": {},
                            "deleted": False, 
                            "milliseconds": 0,
                            "events": [],
                            "regenerate_thumbnail": False
                        }
                    ], 
                    "categories": [
                        {
                            "id": 1,
                            "name": "Ship",
                            "supercategory": "Ship",
                            "color": "#ffffff",
                            "metadata": {},
                            "keypoint_colors": ["#d99100", "#4d8068", "#0d2b80"],
                            "keypoints": ["Bow", "SternLeft", "SternRight"],
                            "skeleton": [[1, 2], [1, 3], [2, 3]]
                        }
                    ],
                    "annotations": [
                        {
                            "id": id,
                            "image_id": image_id,
                            "category_id": id,                                                           # [OUI , ymin, OUI , ymin + ymax, xmin + xmax, ymin + ymax, xmin + xmax, OUI ].
                            "segmentation": [segmentation],  # [xmin, ymin, xmin, ymin + ymax, xmin + xmax, ymin + ymax, xmin + xmax, ymax].
                            "area": area,                         # bbox.width * bbox.height
                            "bbox": bbox,  # [xmin, ymin, width, height] 
                            "iscrowd": False,
                            "isbbox": True,
                            "color": "#e97f2f",
                            #"keypoints": [595, 649, 2, 697, 461, 2, 673, 452, 2], 
                            "keypoints": [int(bateaux[i].avant.x), int(bateaux[i].avant.y), 2, int(bateaux[i].arriere_g.x), int(bateaux[i].arriere_g.y), 2, int(bateaux[i].arriere_d.x), int(bateaux[i].arriere_d.y), 2],
                            "metadata": {},
                            "num_keypoints": 3
                        }
                    ]
                }
                outfile = open(path + "/" + dir_convert_to + "/" + dataset_name, "w")
                json.dump(dictionary, outfile)
                outfile.close()
            else:
                new_boat = {
                    "id": id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": False,
                    "isbbox": True,
                    "color": "#e97f2f",
                    "keypoints": [int(bateaux[i].avant.x), int(bateaux[i].avant.y), 2, int(bateaux[i].arriere_g.x), int(bateaux[i].arriere_g.y), 2, int(bateaux[i].arriere_d.x), int(bateaux[i].arriere_d.y), 2],
                    "metadata": {},
                    "num_keypoints": 3
                }
                add_json_boat(new_boat)
    

        if(image_id > 1):
            new_image = {
                    "id": image_id,
                    "dataset_id": 1,
                    "category_ids": [],
                    "path": path_image,
                    "width": 768,
                    "height": 768,
                    "file_name": file_name,
                    "annotated": False,
                    "annotating": [],
                    "num_annotations": 0,
                    "metadata": {},
                    "deleted": False, 
                    "milliseconds": 0,
                    "events": [],
                    "regenerate_thumbnail": False
                }
            add_json_images(new_image)

        if(os.path.exists(path_dataset_to_copy + "/" + file_name)):
            print("Found")
            shutil.copyfile(path_dataset_to_copy + "/" + file_name, path  + "/" + dir_convert_to + "/" + "images/" + file_name)
        else:
            print("Not Found")

        print()

    return 0


def add_json_images(new_image):

    f_json = open(path + "/" + dir_convert_to + "/" + "dataset.json","r+")
    file_data = json.load(f_json)
    file_data["images"].append(new_image)
    f_json.seek(0)
    json.dump(file_data, f_json)
    f_json.close()

def add_json_boat(new_boat):

    f_json = open(path + "/" + dir_convert_to + "/" + "dataset.json","r+")
    file_data = json.load(f_json)
    file_data["annotations"].append(new_boat)
    f_json.seek(0)
    json.dump(file_data, f_json)
    f_json.close()



print ("-------------------- Convert YOLO V7 to COCO --------------------")

path = os.path.dirname(__file__)

path_dataset_to_copy = path
path_dataset_to_copy = os.path.abspath(os.path.join(path_dataset_to_copy, os.pardir)) # parent folder
path_dataset_to_copy = os.path.abspath(os.path.join(path_dataset_to_copy, os.pardir)) # parent folder
path_dataset_to_copy = os.path.join(path_dataset_to_copy, "annotation_automatique", "input", "dataset") # find the dataset in the folders

print(path)
print(path_dataset_to_copy)

dir_convert_from = "input_YOLO"
dir_convert_to   = "output_COCO"
dataset_name = "dataset.json"

read_all_files(dir_convert_from)