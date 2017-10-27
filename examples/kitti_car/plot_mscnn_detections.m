%% Helps to plot and debug detections
% Reads the long list files of detections from run_mscnn_detection or
% finish detection after server and displays corresponding images, g.t.
% (green) and found objects (red). Furthermore adds likelyhood of each
% detection. Detection with < 20 % likelihodd are not ploted to reduce
% clutter

%% read detections
detections = dlmread('detections/detections_mscnn-7s-384_mscnn_kitti_train_2nd_iter_25000.caffemodel.txt');
%detections(:,1) = detections(:,1)+1;

%% read groundtruth
kitti_dir = '/home/matthias/data/KITTI/';
gt_list = dir([kitti_dir 'training/label_2/*.txt']);

%% set KITTI dataset directory
image_dir = [kitti_dir 'training/image_2/'];
image_list = dir([image_dir '*.png']);  %An object that represents all pictures
nImg=length(image_list);



go_on = true;
i = 1;
while(go_on)
    %% Show Picture
    f = figure;
    hold on;
    imshow([image_list(i).folder '/' image_list(i).name]);

    %% Show detection
    threshold = 0.2;
    det_here = detections(detections(:,1)==i,:); %get boxes of picture i
    det_here = det_here(det_here(:,6)>threshold, :);
    for j = 1:size(det_here,1)
        box = det_here(j,:);    % #pic, x, y, width, height, propability
        if(box(6) > 0.2)
            rectangle('Position', [box(2:5)],'EdgeColor',[1 0 0] * box(6));
            text(box(2),box(3)-15,[num2str(box(6)*100, 3)],'Color','red','FontSize',20);
        end
    end
    
    %% Show Groundtruth
    gt = dlmread([gt_list(i).folder '/' gt_list(i).name],' ',0,1); 
    whereCar = fileread([gt_list(i).folder '/' gt_list(i).name]);   %Find out what are cars
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