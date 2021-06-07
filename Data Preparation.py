#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import glob
import tqdm.notebook as tqdm
import matplotlib.pyplot as plt
import pickle as pkl

im_w, im_h = 128, 128

label_obstacle, label_angleness, label_centerness = [], [], []
images = []
for im_path in tqdm.tqdm(glob.glob('./images/*')):
    im_name = im_path[im_path.rfind('/')+1:im_path.rfind('_')]
    
    angleness = float(im_name[im_name.find('_')+1:im_name.rfind('_')])
    centerness = float(im_name[im_name.rfind('_')+1:])
    
    obstacle = 1 if centerness != 0 else 0
    
    label_obstacle.append(obstacle)
    label_angleness.append(angleness)
    label_centerness.append(centerness)
    
    im = cv2.imread(im_path)
    im = cv2.resize(im, (im_w, im_h), interpolation = cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    images.append(im)


images = np.array(images)
label_obstacle = np.array(label_obstacle)
label_angleness = np.array(label_angleness)
label_centerness = np.array(label_centerness)

# randomly remove 50% samples with no obstacle
positive_ims = np.where(label_obstacle == 0)[0]
keep_idx = np.random.choice(positive_ims, size=int(0.5 * len(positive_ims)), replace=False)

negative_ims = np.where(label_obstacle == 1)[0]
keep_idx = np.concatenate([negative_ims, keep_idx])


images = images[keep_idx]
label_obstacle = label_obstacle[keep_idx]
label_angleness = label_angleness[keep_idx]
label_centerness = label_centerness[keep_idx]

d = {'images': images, 'obstacle': label_obstacle, 'angleness': label_angleness, 'centerness': label_centerness}
with open("data.pkl", 'wb') as pfile:
    pkl.dump(d, pfile, protocol=4)

