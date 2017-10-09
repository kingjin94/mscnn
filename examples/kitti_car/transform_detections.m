%% Takes a table of detection and saves them individualy
% Input: ImgNum x y width height Propability
%        place to store stuff in
% Output in individual files with format
% Car -1 -1 -10 left top right bottom -1 -1 -1 -1000 -1000 -1000 -10 Propability*1000

function transform_detections(detections, dir) 
    num =  max(detections(:,1));
    kitti_dir = '/home/matthias/data/KITTI/';
    addpath([kitti_dir 'devkit/matlab/']);
    %cd dir
    mkdir(dir, 'data/')
    for(i = 1:num)
        in_this_file = detections(detections(:,1) == i,:);
        
        in_this_file(:,4) = in_this_file(:,4) + in_this_file(:,2);
        in_this_file(:,5) = in_this_file(:,5) + in_this_file(:,3);
        
        objects=[]; num = 0;
    
        for j = 1:size(in_this_file,1)
            num = num+1;
            objects(num).type = 'Car'; objects(num).score = in_this_file(j,6)*1000;
            objects(num).x1 = in_this_file(j,2); objects(num).y1 = in_this_file(j,3);
            objects(num).x2 = in_this_file(j,4); objects(num).y2 = in_this_file(j,5);
        end
        
        writeLabels(objects, [dir 'data/'],i-1);
    end
end