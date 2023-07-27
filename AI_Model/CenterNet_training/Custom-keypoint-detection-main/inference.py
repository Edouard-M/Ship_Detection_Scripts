#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 23:28:19 2022

@author: a975193
"""

import cv2
import time
import glob
import numpy as np
import tensorflow.compat.v1 as tf

input_folder = "dataset/images/"
model_path = "saved_model/saved_model"
output_folder = "output_folder/"

colors = [(255,0,0), (229, 52, 235), (235, 85, 52),
          (14, 115, 51), (14, 115, 204)]

cv2.namedWindow("display", cv2.WINDOW_NORMAL)

def process_keypoint(kp, kp_s, h, w, img):
    for i, kp_data in enumerate(kp):
        cv2.circle(img,(int(kp_data[1]*w), int(kp_data[0]*h)),5,colors[i],-1)
    return img

with tf.Session(graph = tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['serve'], model_path)
    graph = tf.get_default_graph()
    input_tensor = graph.get_tensor_by_name("serving_default_input_tensor:0")
    det_score = graph.get_tensor_by_name("StatefulPartitionedCall:6")
    det_class = graph.get_tensor_by_name("StatefulPartitionedCall:2")
    det_boxes = graph.get_tensor_by_name("StatefulPartitionedCall:0")
    det_numbs = graph.get_tensor_by_name("StatefulPartitionedCall:7")
    det_keypoint = graph.get_tensor_by_name("StatefulPartitionedCall:4")
    det_keypoint_score = graph.get_tensor_by_name("StatefulPartitionedCall:3")
    print("Model Loaded")
    for image_path in glob.glob(input_folder+"*.jpg"):
        filename = image_path.split("/")[-1]
        frame = cv2.imread(image_path)
        if frame is not None:
            frame = cv2.resize(frame,(512,512),interpolation = cv2.INTER_AREA)
            height, width, _ = frame.shape
            image_exp_dims = np.expand_dims(frame, axis=0)
            start_time = time.time()
            score,classes,boxes,nums_det, \
            keypoint,keypoint_score = sess.run([det_score, det_class, det_boxes, 
                                                det_numbs,det_keypoint,det_keypoint_score], 
                                                feed_dict={input_tensor:image_exp_dims})
            for i in range(int(nums_det[0])):
                if(score[0][i]*100 > 50): 
                    per_box = boxes[0][i]
                    y1 = int(per_box[0]*height)
                    x1 = int(per_box[1]*width)
                    y2 = int(per_box[2]*height)
                    x2 = int(per_box[3]*width)
        
                    p1 = (x1,y1)
                    p2 = (x2,y2)
                    cv2.rectangle(frame, p1, p2, (0,255,0), 3)
            frame = process_keypoint(keypoint[0][0], keypoint_score[0], height, width, frame)
            cv2.imshow("display",frame)
            cv2.imwrite(output_folder+filename, frame)
            print("Time: ", time.time() - start_time)
            cv2.waitKey(1)
        else:
            print("break")
            break
cv2.destroyAllWindows()