In the "project_scripts" folder are the files useful for training the Centernet model, specifically modified for training keypoints for boats. 


---------------------------------------------project_scripts description -------------------------------------------------------------------

Note that the code for the training has been adapted to work on google colab. Some lines and parameters will have to be adapted according to the environment. 

pipeline.config:
This file allows you to set up the trainings.
The most important parameters are in the lines: 67, 68, 122, 123, 131, 133, 135, 175, 177, 179

label_map.pbxt:
This file contains the information about the annotation. You have to be very careful with the name of the annotations

inference.py:
This file allows to visualize the prediction of the model on test images. 
The paths to the files are at the top of the code and the parameters.
The folder yolo_folder was added during the project.
Its purpose is to store the results of the predictions in yolo format files in order to compare some annotations with the predictions. 

inference_video.py:
Same purpose as inference.py but on ".mp4" video

generate_tfrecord_from_coco:
The file is used to generate the tfrecords for the training from the coco annotations. It has been adapted to the project line 89 and 93. 


All the files modified in this folder are based on the information contained in the prabhakar-sivanesan readme (path:"Custom-keypoint-detection-main\README")
This file contains more information on how to modify the files. The information can also be found on the GitHub directory of the project. ("https://github.com/prabhakar-sivanesan/Custom-keypoint-detection")
--------------------------------------------------------------------------------------------------------------------------------------------

The folder also contains two "saved_model" folders.
These two folders contain the saved weights of the models we have trained.
The folder "saved_model520" contains a model trained on about 520 images.
This dataset is in the folder "dataset_520". The folder "saved_model_1500" contains a model trained on a dataset of 1500 images (folder "dataset_1500").
This dataset has the particularity to have been annotated by hand for 500 images and the 1000 others have been done automatically.
Some images are not perfectly annotated and the goal was to compare the results with a model only trained with hand annotated images.

--------------------------------------------------------------------------------------------------------------------------------------------

We also placed in the "output_folder_*" folders the prediction results of the models we trained