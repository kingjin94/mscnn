%% read detections
detections_list = dir('/home/matthias/mscnn-master/examples/kitti_result/val/kitti_7s_384_25k_car_val/data/*.txt');

%% read groundtruth
kitti_dir = '/home/matthias/data/KITTI/';
gt_list = dir([kitti_dir 'training/label_2/*.txt']);

%% set KITTI dataset directory
image_dir = [kitti_dir 'training/image_2/'];
image_list = dir([image_dir '*.png']);  %An object that represents all pictures
nImg=length(detections_list);



go_on = true;
k = 1;
while(k<=length(detections_list))
    i = str2num(detections_list(k).name(1:6))+1;
    %% Show Picture
    f = figure;
    title(['Picture: ' image_list(i).name]);
    hold on;
    imshow([image_list(i).folder '/' image_list(i).name]);

    %% Show detection if there are any
    tmp = ['Detections from file: ' detections_list(k).name '\n'];
    fprintf(tmp);
    threshold = 0.2;
    if(~strcmp(fileread([detections_list(k).folder '/' detections_list(k).name]),''))
        det_here = dlmread([detections_list(k).folder '/' detections_list(k).name],' ',0,1); %get boxes of picture i
        det_here = det_here(:,[4:7 15]);
        %det_here = det_here(det_here(:,6)>threshold, :);
        det_here = [det_here(:,1) det_here(:,2) det_here(:,3)-det_here(:,1) det_here(:,4)-det_here(:,2) det_here(:,5)];
        for j = 1:size(det_here,1)
            box = det_here(j,:);    % #pic, x, y, width, height, propability
            if(1)%box(6) > 0.2)
                rectangle('Position', [box(1:4)],'EdgeColor',[1 0 0] * box(5)/1000);    %Propability scaled by 1000
            end
        end
    end
    
    %% Show Groundtruth 
    tmp = ['GT from file: ' gt_list(i).name '\n'];
    fprintf(tmp);
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
    k = k+1;
    if(k>nImg) k = 1; end
end