%% Init
% Find all available detections
result_list = dir([pwd '/outputDetection/*-7s-*.tar.gz']);

% Find img sizes
if exist('train_sizes.txt')
    sizes = dlmread('train_sizes.txt');
else
    image_size;
end

imageSet = fullfile(pwd, '../../data/DitM/ImageSets_day/val.txt')

for i = 1:size(result_list,1)
    tmpResult = finish_detection_after_server([result_list(i).folder '/' result_list(i).name], sizes, ...
        dlmread(imageSet));
    indexSlash = regexp(tmpResult,'[/]');
    lastSlash = indexSlash(end);
    evalFunc(tmpResult(lastSlash+1:end), imageSet);
    close all
end
