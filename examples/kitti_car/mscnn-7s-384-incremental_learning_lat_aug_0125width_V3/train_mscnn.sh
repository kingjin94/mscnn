GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_1_1st.prototxt \
  --weights=../../../models/VGG/VGG_ILSVRC_16_layers.caffemodel \
  --gpu=0  2>&1 | tee log_1_1st.txt

GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_2_1nd.prototxt \
  --weights=mscnn_kitti_train_1_1st_iter_6000.caffemodel \
  --gpu=0  2>&1 | tee log_2_1nd.txt

GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_1_2st.prototxt \
  --weights=mscnn_kitti_train_2_1nd_iter_10000.caffemodel \
  --gpu=0  2>&1 | tee log_1_2st.txt

GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_2_2nd.prototxt \
  --weights=mscnn_kitti_train_1_2st_iter_6000.caffemodel \
  --gpu=0  2>&1 | tee log_2_2nd.txt
