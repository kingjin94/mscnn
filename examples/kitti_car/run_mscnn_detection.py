## Import
# Misc
import numpy as np
import matplotlib.pyplot as plt

# Caffe
import sys
caffe_root = '../../'  # this file should be run from {caffe_root}/examples/kitti_car
sys.path.insert(0, caffe_root + 'python')

import caffe

## Start Caffe
# Find and load model
import os
model_def = caffe_root + 'examples/kitti_car/mscnn-7s-384/mscnn_deploy.prototxt'	# add deployable net here
model_weights = caffe_root + 'examples/kitti_car/mscnn-7s-384/*.caffemodel'			# add trained model here
if os.path.isfile(caffe_root + 'examples/kitti_car/mscnn-7s-384/mscnn_deploy.prototxt'):
    print 'CaffeNet found.'
else:
    print 'No net found'
    return

# Init Caffe
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
                
net.forward()  # run once before timing to set up memory

## Set KITTI dataset directory
kitti_root = '/home/data/KITTI/'
image_dir = [kitti_root 'training/image_2/']; 	# gt only available for training set!!!
comp_id = 'test';   							# CHANGE FOR OTHER MODEL
image_list = dir([image_dir '*.png']);  		# An object that represents all pictures
nImg=length(image_list);
imgW = 1280; imgH = 384;
