%% read detections
detections = dlmread('detections/detectionsOnDitM_mscnn-7s-384_mscnn_kitti_train_2nd_iter_25000.caffemodel.txt');
%detections(:,1) = detections(:,1)+1;

%% read groundtruth
kitti_dir = '/home/matthias/Desktop/GTA_KITTI/';
gt_dir = '../../data/DitM/label_2/';

%% read dataset
image_list = dlmread('../../data/DitM/ImageSets_small/val.txt');

%% set KITTI dataset directory
image_dir = [kitti_dir 'training/image_2/'];
%image_list = dir([image_dir '*.jpg']);  %An object that represents all pictures
nImg=5000; %length(image_list);



go_on = true;
i = 1;
while(go_on)
    %% Show Picture
    f = figure;
    hold on;
    imshow([image_dir num2str(image_list(i),7) '.jpg']);
    disp [image_dir num2str(image_list(i),7) '.jpg']
    
    %% Show detection
    threshold = 0.2;
    det_here = detections(detections(:,1)==i,:); %get boxes of picture i
    det_here = det_here(det_here(:,6)>threshold, :);
    for j = 1:size(det_here,1)
        box = det_here(j,:);    % #pic, x, y, width, height, propability
        if(box(6) > 0.2)
            rectangle('Position', [box(2:5)],'EdgeColor',[1 0 0] * box(6));
        end
    end
    
    %% Show Groundtruth
    gt = dlmread([gt_dir num2str(image_list(i),7) '.txt'],' ',0,1); 
    whereCar = fileread([gt_dir num2str(image_list(i),7) '.txt']);   %Find out what are cars
    whereCar = regexp(whereCar,'[ \n]','split');
    whereCar = whereCar(1:15:end);
    whereCar = strcmp(whereCar, 'Car');
    gt = gt(whereCar,:);    %Only disp cars
    %Structure: type|truncated|occluded|alpha|bbox⁴|...
    %bbox: left, top, right, bottom in pixel
    gt = gt(:,4:7); %only take bbox
    gt = [gt(:,1) gt(:,2) gt(:,3)-gt(:,1) gt(:,4)-gt(:,2)];
    for j = 1:size(gt,1)
        rectangle('Position', [gt(j,:)],'EdgeColor','g')
    end
    
    if(size(gt,1) == 1 && size(det_here,1) == 1)
        IoU = bboxOverlapRatio(det_here(2:5), gt)
    end
    
    waitfor(f)
    i = i+1
    if(i>nImg) i = 1; end
end