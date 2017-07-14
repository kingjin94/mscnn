%% Change format to KITTI standard
run_results = 'mscnn-7s-384-train_lateral_augment_0125width_Layer4_car.txt'
detections = dlmread(['detections/' run_results]); %Results from run_mscnn_detection_backup.m
result_dir = '/home/matthias/mscnn-master/examples/kitti_car/detections/mscnn-7s-384-results_lateral_augment_width0125_Layer4_train/';
transform_detections(detections, result_dir);

%% Evaluate with KITTI (only for training set...)
kitti_dir = '/home/matthias/data/KITTI/';
gt_dir = [kitti_dir 'training/label_2/'];
list_dir = ['/home/matthias/mscnn-master/data/kitti/ImageSets/val.txt'];

command_line = sprintf('/home/matthias/mscnn-master/examples/kitti_result/eval/evaluate_object %s %s %s',...
        gt_dir,result_dir,list_dir);
system(command_line);

results = dlmread([result_dir 'plot/car_detection.txt']);

figure
hold on
plot(results(:,1),results(:,2:4))
legend('easy','medium','hard')
xlabel('Recall')
ylabel('Precision')
title(['Results from ' run_results]);
