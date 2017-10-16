tag = 'detections_mscnn-7s-384-incremental_learning_lat_aug_025width_mscnn_kitti_train_2';
timeSeries = precisionRecallOverTime(tag);

%% Precision Recall over time
f = figure
set(f,'PaperOrientation','landscape');

hold on

%colormap('gray')

subplot(1,4,1)
surf([2000:2000:24000 25000],timeSeries(1,:,1),timeSeries(:,:,2)','EdgeColor','none')
view(0,90)
title('easy')
xlabel('Itteration')
ylabel('Recall')
%zlabel('Precision')
axis tight
pbaspect([1 1 1])

subplot(1,4,2)
surf([2000:2000:24000 25000],timeSeries(1,:,1),timeSeries(:,:,3)','EdgeColor','none')
view(0,90)
title('medium')
xlabel('Itteration')
ylabel('Recall')
%zlabel('Precision')
axis tight
pbaspect([1 1 1])

subplot(1,4,3)
surf([2000:2000:24000 25000],timeSeries(1,:,1),timeSeries(:,:,4)','EdgeColor','none')
view(0,90)
title('hard')
xlabel('Itteration')
ylabel('Recall')
%zlabel('Precision')
axis tight
pbaspect([1 1 1])

subplot(1,4,4)

axis off

c = colorbar('Location','westoutside');
c.Label.String = 'Precision';
pbaspect([1 2 1])

ha = axes('Xlim',[0 1],'Ylim',[0  1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
text(-0.1, 0.98,['Recall Precision over training for ' tag], 'Interpreter', 'none')

%% Avg Precision recall over time
f = figure

title(['AP over training for ' tag], 'Interpreter', 'none')

avg = zeros(13,3);
for j = 1:13
    results = squeeze(timeSeries(j,:,:));
    
    % Calc precision according to Pascal VOC
    
    for i = 0:10
        avg(j,:) = avg(j,:) + max(results(i*4+1:end,2:4));
    end
end
avg = avg/11;

hold on
plot([2000:2000:24000 25000], avg(:,1))
plot([2000:2000:24000 25000], avg(:,2))
plot([2000:2000:24000 25000], avg(:,3))
legend('Easy', 'Medium', 'Hard')