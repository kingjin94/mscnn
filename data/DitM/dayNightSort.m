% Sorts Images of DitM according to the mean V (from Image in HSV space) to sort day and night images. Results are saved to daySet.txt and nightSet.txt respectively

%% Get avg brigthness in rgb
pics = dir([pwd '/VOC2012/JPEGImages/*.jpg']);
%pics = dir('/home/matthias/Desktop/GTA_10k/VOC2012/JPEGImages/*.jpg')

len = length(pics);

brightness = zeros(3,length(pics));

progress = 0;

parfor i = 1:length(pics)
    pic = imread([pics(i).folder '/' pics(i).name]);
    brightness(:,i) = squeeze(mean(mean(pic)));

    if(mod(i,500) == 0)
        progress = progress + 500;
        disp([num2str(progress) '/' num2str(len)]);
    end
end

%% Sort images
day = zeros(1,length(pics));
night = zeros(1,length(pics));
for i = 1:length(pics)
    hsv = rgb2hsv(brightness(:,i)'/255);
    
    if(hsv(3) > 0.3) % Determine if Day or night
        %copyfile([pics(i).folder '/' pics(i).name], ['day/' pics(i).name]);
        str = pics(i).name;
        day(i) = str2num(str(1:7));
    else
        %copyfile ([pics(i).folder '/' pics(i).name], ['night/' pics(i).name]);
        str = pics(i).name;
        night(i) = str2num(str(1:7));
    end
    
    if(mod(i,100) == 0)
       disp([num2str(i) '/' num2str(length(pics))]) 
    end
end

dlmwrite('daySet.txt', day(day~=0)', 'precision',7)
dlmwrite('nightSet.txt', night(night~=0)', 'precision',7)
