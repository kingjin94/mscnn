% set KITTI dataset directory
root_dir = '/home/matthias/data/KITTI/';
image_dir = [root_dir 'training/image_2/']; %gt only available for training set!!!
comp_id = 'mscnn-7s-384-lateral_augment_0125width';   %CHANGE FOR OTHER MODEL
image_list = dir([image_dir '*.png']);  %An object that represents all pictures
nImg=length(image_list);

sizes = zeros(nImg,2);
%% Loop over all images
for k = 1:nImg
    %% Load and prepare Image
    test_image = imread([image_dir num2str(k-1, '%06d') '.png']);
    [sizes(k,1),sizes(k,2),~] = size(test_image);
    
    if (mod(k,100)==0)
        fprintf('idx %i/%i\n',k,nImg);
    end
end

%% Save sizes 
dlmwrite('train_sizes.txt',sizes);