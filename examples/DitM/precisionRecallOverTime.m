%% format: [step,recall,{recall level, easy, medium, hard}]

function [recallPrec] = precisionRecallOverTime(prefix)
    results = dir(['detections/' prefix '*/plot/car_detection.txt']);

    tmp = cell(3,size(results,1));
    for i = 1:size(results,1)
        tmp{1,i} = results(i).folder;
        tmp{3,i} = dlmread([results(i).folder '/car_detection.txt']);
        run = regexp(results(i).folder,'(_2_1|_2_2|_2)nd','match');
        run = regexprep(run,'\D',''); %rm non digits
        itter = regexp(results(i).folder,'[1-9]+0000?','match');
        tmp{2,i} = str2num(run{1})*10^6 + str2num(itter{1});
    end
    [~,I] = sort(cell2mat(tmp(2,:)));
    tmp = tmp(:,I);
    %Extract surface
    recallPrec = zeros(size(results,1),length(tmp{3,1}),4);
    for i = 1:size(results,1)
        recallPrec(i,:,:) = tmp{3,i};
    end
end