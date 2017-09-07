
#GLOG_logtostderr=1 ../../../build/tools/caffe train \
#  --solver=solver_1st.prototxt \
#  --weights=../../../models/VGG/VGG_ILSVRC_16_layers.caffemodel \
#  --gpu=0,1,2,3  2>&1 | tee log_1st.txt

GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_2nd.prototxt \
  --weights=mscnn_kitti_train_2nd_iter_25000.caffemodel \
  --gpu=4,5,6,7  2>&1 | tee log_2nd.txt
