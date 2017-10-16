% Copyright (c) 2017 TUM
% see mscnn/LICENSE for details
% Written by Matthias Mayer 
% Input:
%  - tar to be analyses
%  - img Sizes
%  - imgNums: the IDs to be analysed

function result = finish_detection_after_server(tarFile,imgSizes,imgNums)
%% Preperation
% Add needed paths
addpath('../../utils/');    % Has Maxsuppresion code

nImg=length(imgNums);

%% Untar preliminary results from python
resultDir = '/tmp/finish_detection/'; %tmp storage
disp 'Untar'
system(['rm -r ' resultDir]);
system(['mkdir ' resultDir]);
system(['tar xzf ' tarFile ' -C /tmp/finish_detection/ --strip-components=2']); %Normally 2!!!
disp 'done, begin work'
indexSlash = regexp(tarFile,'[/]');
lastSlash = indexSlash(end);
mkdir('detections');
resultFile = [pwd '/detections/' tarFile(lastSlash+1:end-7) '.txt'];

%% Model params
% choose the right input size -> modeldependent
imgW = 1280; imgH = 384;
% imgW = 1920; imgH = 576;
% imgW = 2560; imgH = 768;

% bbox de-normalization parameters
bbox_means = [0 0 0 0];
bbox_stds = [0.1 0.1 0.2 0.2];

% non-maximum suppression parameters
pNms.type = 'maxg'; pNms.overlap = 0.5; pNms.ovrDnm = 'union';

cls_ids = [2]; num_cls=length(cls_ids); %Car or not car are the classes
obj_names = {'bg','car','van','truck','tram'};
final_detect_boxes = cell(100,num_cls); final_proposals = cell(100,1);
proposal_thr = -10; usedtime=0;

show = 0; % Kills memory -> only use with some pics, not all at once
show_thr = 0.1;

%% Loop over all images
for k = 1:nImg
    %% Load Image sizes and ratio
    orgH = imgSizes(k,1);
    orgW = imgSizes(k,2);
    % Change for other dataset
    ratios = [imgH imgW]./[orgH orgW];
    
    %% Read server results
    outputs = {dlmread([resultDir 'bbox_preds_' num2str(imgNums(k),7) '.txt'])', ...
        dlmread([resultDir 'cls_pred_' num2str(imgNums(k),7) '.txt'])', ...
        dlmread([resultDir 'tmp_' num2str(imgNums(k),7) '.txt'])'};
    

    
    %% Work on Output -> Seperate vectors
    tmp=squeeze(outputs{1}); bbox_preds = tmp'; %Bounding boxes?
    tmp=squeeze(outputs{2}); cls_pred = tmp';   %Classes?
    tmp=squeeze(outputs{3}); 
    tmp = tmp'; tmp = tmp(:,2:end);
    tmp(:,3) = tmp(:,3)-tmp(:,1); tmp(:,4) = tmp(:,4)-tmp(:,2);
    proposal_pred = tmp; proposal_score = proposal_pred(:,end);
    
    % filtering some bad proposals -> confidence higher than threshold and
    % both dims bigger 0
    keep_id = find(proposal_score>=proposal_thr & proposal_pred(:,3)~=0 & proposal_pred(:,4)~=0);
    proposal_pred = proposal_pred(keep_id,:);
    bbox_preds = bbox_preds(keep_id,:); 
    cls_pred = cls_pred(keep_id,:);
    
    %Normalize to resize?!
    proposals = double(proposal_pred);
    proposals(:,1) = proposals(:,1)./ratios(2);
    proposals(:,3) = proposals(:,3)./ratios(2);
    proposals(:,2) = proposals(:,2)./ratios(1);
    proposals(:,4) = proposals(:,4)./ratios(1);
    final_proposal = proposals;
    
    %% Per class some display stuff?
    for i = 1:num_cls
        id = cls_ids(i); bbset = [];
        bbox_pred = bbox_preds(:,id*4-3:id*4);
        
        % bbox de-normalization
        bbox_pred = bbox_pred.*repmat(bbox_stds,[size(bbox_pred,1) 1]);
        bbox_pred = bbox_pred+repmat(bbox_means,[size(bbox_pred,1) 1]);
        
        exp_score = exp(cls_pred);
        sum_exp_score = sum(exp_score,2);
        prob = exp_score(:,id)./sum_exp_score;
        ctr_x = proposal_pred(:,1)+0.5*proposal_pred(:,3);
        ctr_y = proposal_pred(:,2)+0.5*proposal_pred(:,4);
        tx = bbox_pred(:,1).*proposal_pred(:,3)+ctr_x;
        ty = bbox_pred(:,2).*proposal_pred(:,4)+ctr_y;
        tw = proposal_pred(:,3).*exp(bbox_pred(:,3));
        th = proposal_pred(:,4).*exp(bbox_pred(:,4));
        tx = tx-tw/2; ty = ty-th/2;
        tx = tx./ratios(2); tw = tw./ratios(2);
        ty = ty./ratios(1); th = th./ratios(1);
        
        % clipping bbs to image boarders
        tx = max(0,tx); ty = max(0,ty);
        tw = min(tw,orgW-tx); th = min(th,orgH-ty);
        bbset = double([tx ty tw th prob]);
        idlist = 1:size(bbset,1); bbset = [bbset idlist'];
        bbset=bbNms(bbset,pNms);
        final_detect_boxes{i} = bbset(:,1:5);
        
        if (show)
            proposals_show = zeros(0,5); bbs_show = zeros(0,6);
            if (size(bbset,1)>0)
                show_id = find(bbset(:,5)>=show_thr);
                bbs_show = bbset(show_id,:);
                proposals_show = proposals(bbs_show(:,6),:);
            end
            % proposal
            for j = 1:size(proposals_show,1)
                rectangle('Position',proposals_show(j,1:4),'EdgeColor','g','LineWidth',2);
                show_text = sprintf('%.2f',proposals_show(j,5));
                x = proposals_show(j,1)+0.5*proposals_show(j,3);
                text(x,proposals_show(j,2),show_text,'color','r', 'BackgroundColor','k','HorizontalAlignment',...
                    'center', 'VerticalAlignment','bottom','FontWeight','bold', 'FontSize',8);
            end
            % detection
            for j = 1:size(bbs_show,1)
                rectangle('Position',bbs_show(j,1:4),'EdgeColor','y','LineWidth',2);
                show_text = sprintf('%s=%.2f',obj_names{id},bbs_show(j,5));
                x = bbs_show(j,1)+0.5*bbs_show(j,3);
                text(x,bbs_show(j,2),show_text,'color','r', 'BackgroundColor','k','HorizontalAlignment',...
                    'center', 'VerticalAlignment','bottom','FontWeight','bold', 'FontSize',8);
            end
        end
    end
    
    % Save stuff
    for j=1:num_cls
        final_detect_boxes{j}=[ones(size(final_detect_boxes{j},1),1)*imgNums(k) final_detect_boxes{j}];  %Format: img_num tx ty tw th prob
    end
    %final_proposal = [ones(size(final_proposal,1),1) final_proposal];
    
    for j=1:num_cls
        id = cls_ids(j);
        save_detect_boxes=cell2mat(final_detect_boxes(:,j));
        if(k == 1)
            dlmwrite(resultFile,save_detect_boxes,'precision',7);
        else
            dlmwrite(resultFile,save_detect_boxes,'-append','precision',7); 
        end
    end
    
    if (mod(k,100)==0)
        fprintf('idx %i/%i\n',k,nImg);
    end
end
% Clean up tmp
rmdir '/tmp/finish_detection' s
result = resultFile;
end
