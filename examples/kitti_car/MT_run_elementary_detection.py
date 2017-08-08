# Evaluation script for MSCNN models
# To be run  in /examples/kitti_car/
# specify Name(!) of dir with .caffemodel to be evaluated
# specify which caffemodel to load

import os as os #paths
import sys

if not len(sys.argv) == 3:
    print 'Specify modelDir and model to evaluate'
    quit()

modelDir = os.path.abspath(sys.argv[1])
if not os.path.exists(modelDir):
    print 'Specify valid model Dir'
    quit()

model = sys.argv[2]


import numpy as np
import numpy.matlib
import re #regex
import scipy.misc #imresize...
import threading
import Queue
import time

# Init Caffe
caffe_root = '/home/mscnn/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0) #Which gpu if there are multiple

# Global Vars
mu = np.array([104,117,123]) # Color offset
imgW = 1280; imgH = 384; # Size of pictures while training
#modelDir = 'mscnn-7s-384'
outputDir = '/tmp/detections_' + str(time.strftime("%H_%M_%S")) + '/'
# Set up place to store results
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# Set up data - May have to change things here
data_root = '/home/data/KITTI/'
image_dir = os.path.join(data_root, 'training/image_2/')
assert os.path.exists(image_dir)
comp_id = 'test'
image_list = os.listdir(image_dir)
only_png = re.compile(r'.png')
image_list = filter(only_png.search, image_list) # List of all images
nImg = len(image_list)

# Workers
numPrep = 4
numGPU = 2
numPost = 4

# Queue work
img_queue = Queue.Queue(100)
preped_img_queue = Queue.Queue(100)
output_queue = Queue.Queue(100)

def prepPic():#img):
    while True:
        img = img_queue.get()
        if isinstance(img, str) and img == 'quit':
            break
        #print 'Processing: ' + img
    
    # Load and preprocess
        test_image = np.array(caffe.io.load_image(image_dir + img, color=True)).squeeze()
        test_image = scipy.misc.imresize(test_image,(imgH,imgW), interp='bicubic')
        test_image = test_image[:,:,(2,1,0)] #Rearange color to BGR
        test_image = test_image.astype(np.float32, copy=False) # to single precision
        test_image -= mu
        test_image = np.transpose(test_image, (2,0,1)) # rearange dim order to width, height color 
    
    
    # Find image id
        id_filter = re.compile(r"[0-9]")
        ID = filter(id_filter.search, img) # List of all images
        preped_img_queue.put({'img':test_image,'id':ID})
        img_queue.task_done()

def GPU_Worker():
    # Set up model to be evaluated
    _model_weights = os.path.join(modelDir, model)#'mscnn_kitti_train_2nd_iter_25000.caffemodel')
    assert os.path.exists(_model_weights)
    _model_definition = os.path.join(modelDir, 'mscnn_deploy.prototxt')
    assert os.path.exists(_model_definition)

    # Get own caffe net
    caffe.set_mode_gpu() #Must be set per thread!
    caffe.set_device(0) #Which gpu if there are multiple
    _net = caffe.Net(_model_definition, _model_weights, caffe.TEST)


    while True:
        #print 'GPU on standby'
        content = preped_img_queue.get()
        if isinstance(content, str) and content == 'quit':
            break
        # Forward image through net
        #print 'GPU working'
        _net.blobs['data'].data[...] = content['img']
        outputs = _net.forward()

        output_queue.put({'output':outputs, 'id':content['id']})
        preped_img_queue.task_done()

    print 'GPU_Worker done'

def Storage_Worker():

    while True:
        #print 'Storage on standby'
        content = output_queue.get()
        if isinstance(content, str) and content == 'quit':
            break

        #print 'Storage working'
        ID = content['id']
        outputs = content['output']
        # Postprocess
        bbox_preds = outputs['bbox_pred']
        #print bbox_preds.shape
        cls_pred = outputs['cls_pred']
        #print cls_pred.shape
        tmp = outputs['proposals_score'].squeeze(axis=(2,3))
        # Save for processing in matlab
        np.savetxt(outputDir+'bbox_preds_'+ID+'.txt', bbox_preds)
        np.savetxt(outputDir+'cls_pred_'+ID+'.txt', cls_pred)
        np.savetxt(outputDir+'tmp_'+ID+'.txt', tmp)
        output_queue.task_done()

    print 'Storage_Worker done'

# Set up the threads
print 'Start forwarding and storing'

# Start worker
for i in range(numPrep):
    t = threading.Thread(target=prepPic)
    t.daemon= True
    t.start()

print 'Preprocess all running'
    
for i in range(numGPU):
    t = threading.Thread(target=GPU_Worker)
    t.daemon= True
    t.start()

print 'GPU all running'
    
for i in range(numPost):
    t = threading.Thread(target=Storage_Worker)
    t.daemon= True
    t.start()
    
print 'Postprocess all running'

# Beginn feeding img queue
i = 0
for item in image_list:
    img_queue.put(item)
    if i%100 == 0:
        print str(i)+' of '+str(nImg)
        print 'Pre/GPU/Post ' + str(img_queue.qsize()) + '/' + str(preped_img_queue.qsize()) + '/' + str(output_queue.qsize())

    i += 1


for i in range(numPrep):
    img_queue.put('quit') 

while not img_queue.empty():
    print 'Elements remaining: ' + str(img_queue.qsize())
    print 'Elements preped/post: ' + str(preped_img_queue.qsize()) + '/' + str(output_queue.qsize())
    time.sleep(1)

for i in range(numGPU):
    preped_img_queue.put('quit')

while not preped_img_queue.empty():
    print 'Waiting for GPU'
    time.sleep(1)
    
for i in range(numPost):
    output_queue.put('quit')

while not output_queue.empty():
    print 'Waiting for Storage'
    time.sleep(1)
    
os.system('tar czf /home/detections_' + str(modelDir) + '_' + model + '.tar.gz ' + str(outputDir))
os.system('rm -rf ' + str(outputDir))
print 'Done, all detections saved to: /home/detections_' + modelDir + '.tar.gz'

