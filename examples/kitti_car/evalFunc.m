function evalFunc(run_results)
    %% Change format to KITTI standard
    %test if empty
    file = dir(['detections/' run_results]);
    if file.bytes == 0
        disp(['detections/' run_results ' is empty!'])
        return
    end
    detections = dlmread(['detections/' run_results]); %Results from run_mscnn_detection_backup.m
    result_dir = ['/home/matthias/mscnn-master/examples/kitti_car/detections/' run_results(1:end-4) '/'];
    transform_detections(detections, result_dir);

    %% Evaluate with KITTI (only for training set...)
    kitti_dir = '/home/matthias/data/KITTI/';
    gt_dir = [kitti_dir 'training/label_2/'];
    list_dir = ['/home/matthias/mscnn-master/data/kitti/ImageSets/val.txt'];

    command_line = sprintf('/home/matthias/mscnn-master/examples/kitti_result/eval/evaluate_object %s %s %s',...
            gt_dir,result_dir,list_dir);
    system(command_line);

    %% Build own plot
    results = dlmread([result_dir 'plot/car_detection.txt']);

    % Calc precision according to Pascal VOC
    avg = zeros(1,3);
    for i = 0:10
        avg = avg + max(results(i*4+1:end,2:4));
    end
    avg = avg/11;
    
    figure
    hold on
    plot(results(:,1),results(:,2:4))
    lgd = legend(['easy (' num2str(avg(1)*100,4) ' %)'], ['medium (' num2str(avg(2)*100,4) ' %)'], ['hard (' num2str(avg(3)*100,4) ' %)']);
    title(lgd, 'Difficulty (Mean Precision)')
    legend('Location','southwest')
    xlabel('Recall')
    ylabel('Precision')
    title(['Results from ' run_results], 'Interpreter', 'none');
    print([result_dir 'plot/forLaTeX'],'-dpdf')
    system(['pdfcrop --margins 10 ' result_dir 'plot/forLaTeX.pdf ' result_dir 'plot/forLaTeX.pdf'])
end