import numpy as np
import numpy.matlib
import os as os #paths
import re #regex
import scipy.misc #imresize...

# Get Caffe ready
caffe_root = '/home/mscnn/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

# Set up model to be evaluated
model = 'mscnn-7s-384'
print 'Evaluating model in ' + model
root_dir = os.path.join(caffe_root, 'examples/kitti_car/' + model + '/')
model_weights = os.path.join(root_dir, 'mscnn_kitti_train_2nd_iter_25000.caffemodel')
assert os.path.exists(model_weights)
model_definition = os.path.join(root_dir, 'mscnn_deploy.prototxt')
assert os.path.exists(model_definition)
imgW = 1280; imgH = 384; # Size of pictures while training

# Init caffe
caffe.set_mode_gpu()
caffe.set_device(0) #Which gpu if there are multiple
net = caffe.Net(model_definition, model_weights, caffe.TEST)

# Set up data
data_root = '/home/data/'
image_dir = os.path.join(data_root, 'training/image_2/')
assert os.path.exists(image_dir)
comp_id = 'test'
image_list = os.listdir(image_dir)
only_png = re.compile(r'.png')
image_list = filter(only_png.search, image_list) # List of all images
nImg = len(image_list)

# Set up place to store results
if not os.path.exists('/home/detections_' + model +'/'):
    os.makedirs('/home/detections_' + model +'/')

# Set up filters - normalize color
mu = np.array([104,117,123])

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
    
    np.savetxt('/home/detections_' + model +'/bbox_preds_'+str(ID)+'.txt', bbox_preds)
    np.savetxt('/home/detections_' + model +'/cls_pred_'+str(ID)+'.txt', cls_pred)
    np.savetxt('/home/detections_' + model +'/tmp_'+str(ID)+'.txt', tmp)

print 'Results saved to /home/detections_' + model +'/'
