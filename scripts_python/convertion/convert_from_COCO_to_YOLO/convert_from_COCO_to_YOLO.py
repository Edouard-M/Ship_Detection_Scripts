import os
import json


print ("-------------------- Convert COCO to YOLO --------------------")

path = os.path.dirname(__file__)

dir_convert_from = "input_COCO"
dir_convert_to   = "output_YOLO"

dataset_name = "dataset.json"


f_json = open(path + "/" + dir_convert_from + "/" + dataset_name)
    
data = json.load(f_json)

for image in data['images']:
    keypoints_list = []
    keypoints = []
    file_name = image['file_name']
    file_name = (file_name[0:9] + ".jpg")
    image_id = image['id']

    for annotation in data['annotations']:
        search_image_id = annotation['image_id']
        if(search_image_id == image_id):
            keypoints = annotation['keypoints']
            keypoints_list.append(keypoints)

    print("\n")
    print(file_name)
    print("ID = ", image_id)
    print("Number of boats = ", len(keypoints_list))
    print(keypoints_list)


    fmode = "x"
    if(os.path.exists(path  + "/" + dir_convert_to + "/" + file_name[0:-4] + ".txt")):
        fmode = "w"
    else:
        fmode = "x"

    f = open(path + "/" + dir_convert_to + "/" + file_name[0:-4] + ".txt", fmode)

    for i in range(len(keypoints_list)):
        if(i!=0):
            f.write("\n")
        f.write("0 " + str(keypoints_list[i][0]) + " " + str(keypoints_list[i][1]))  # Bow
        f.write("\n3 " + str(keypoints_list[i][3]) + " " + str(keypoints_list[i][4]))  # Left
        f.write("\n2 " + str(keypoints_list[i][6]) + " " + str(keypoints_list[i][7]))         # Right

    f.close()

f_json.close()