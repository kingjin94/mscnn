# Evaluation script for MSCNN models
# To be run  in /examples/kitti_car/
# specify Name(!) of dir with .caffemodel to be evaluated
# specify which caffemodel to load

import os as os #paths
import sys
import time

if not len(sys.argv) == 3:
    print 'Specify modelDir and model to evaluate'
    quit()

modelDir = os.path.abspath(sys.argv[1])
modelName = sys.argv[1]
if not os.path.exists(modelDir):
    print 'Specify valid model Dir'
    quit()

model = sys.argv[2]

print '------------------------------------------------------------------'
print 'Called with: '
print modelDir
print model
print '------------------------------------------------------------------'
print ''


import numpy as np
import numpy.matlib
import re #regex
import scipy.misc #imresize...

# Set up model to be evaluated
model_weights = os.path.join(modelDir, model)#'mscnn_kitti_train_2nd_iter_25000.caffemodel')
assert os.path.exists(model_weights)
model_definition = os.path.join(modelDir, 'mscnn_deploy.prototxt')
assert os.path.exists(model_definition)

# Init Caffe
os.environ['GLOG_minloglevel'] = '2'
caffe_root = '/home/mscnn/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0) #Which gpu if there are multiple
net = caffe.Net(model_definition, model_weights, caffe.TEST)

# Global Vars
mu = np.array([104,117,123]) # Color offset
imgW = 1280; imgH = 384; # Size of pictures while training
#modelDir = 'mscnn-7s-384'
outputDir = '/tmp/detections_' +modelName+'_'+model+'_'+ str(time.strftime("%H_%M_%S")) + '/'
# Set up place to store results
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# Set up data
data_root = '/home/data/KITTI/'
image_dir = os.path.join(data_root, 'training/image_2/')
assert os.path.exists(image_dir)
comp_id = 'test'
image_list = os.listdir(image_dir)
only_png = re.compile(r'.png')
image_list = filter(only_png.search, image_list) # List of all images
nImg = len(image_list)

# Loop over all pictures
for k in range(0, nImg):
    if k%100==0:
        print 'In Loop ' + str(k)
    #Preprocess - might be faster with imagetransformer by  caffe...
    test_image = np.array(caffe.io.load_image(image_dir + image_list[k], color=True)).squeeze()

    test_image = scipy.misc.imresize(test_image,(imgH,imgW), interp='bicubic')
    test_image = test_image[:,:,(2,1,0)] #Rearange color to BGR
    test_image = test_image.astype(np.float32, copy=False) # to single precision
    #test_image = np.apply_along_axis(mu, 2, test_image) #subtract mean per channel on each pixel
    test_image -= mu
    test_image = np.transpose(test_image, (2,0,1)) # rearange dim order to width, height color 

    # Forward image through net
    net.blobs['data'].data[...] = test_image
    outputs = net.forward()

    # Postprocess
    bbox_preds = outputs['bbox_pred']#.squeeze()
    cls_pred = outputs['cls_pred']#.squeeze()
    tmp = outputs['proposals_score'].squeeze(axis=(2,3))


    # Save for processing in matlab
    id_filter = re.compile(r"[0-9]")
    ID = filter(id_filter.search, image_list[k]) #Find Img ID with regex

    np.savetxt(outputDir+'bbox_preds_'+str(ID)+'.txt', bbox_preds)
    np.savetxt(outputDir+'cls_pred_'+str(ID)+'.txt', cls_pred)
    np.savetxt(outputDir+'tmp_'+str(ID)+'.txt', tmp)

os.system('tar czf /home/output/detections_' + str(modelName) + '_' + str(model) + '.tar.gz ' + str(outputDir))
os.system('rm -rf ' + str(outputDir))
print 'Done, all detections saved to: /home/detections_' + str(modelName) + '_' + str(model) + '.tar.gz '
