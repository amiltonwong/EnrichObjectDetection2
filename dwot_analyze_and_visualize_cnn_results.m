function [ ap ] = dwot_analyze_and_visualize_cnn_results( detection_result_txt, ...
                        detectors, VOCopts, param, nms_threshold, visualize, save_path,...
                        n_views, prediction_azimuth_rotation_direction, prediction_azimuth_offset)
if nargin < 6
    visualize = false;
end

if nargin < 8
    b_compute_view = false;
end

if nargin == 8
    prediction_azimuth_rotation_direction = 1;
    prediction_azimuth_offset = 0;
    b_compute_view = true;
    max_azimuth_difference = 360/n_views/2;
    confusion_statistics = zeros(n_views, n_views);
end

if nargin > 8
    b_compute_view = true;
    max_azimuth_difference = 360/n_views/2;
    confusion_statistics = zeros(n_views, n_views);
end

% renderings = cellfun(@(x) x.rendering_image, detectors, 'UniformOutput', false);

% PASCAL_car_val_init_0_Car_each_7_lim_300_lam_0.0150_a_24_e_2_y_1_f_2_scale_2.00_sbin_6_level_10_nms_0.40_skp_e
% Group inside group not supported in matlab
detection_params = dwot_detection_params_from_name(detection_result_txt);
image_scale_factor = detection_params.scale;

% SNames = fieldnames(temp); 
% for loopIndex = 1:numel(SNames) 
%     stuff = S.(SNames{loopIndex})
% end

skip_criteria = dwot_recover_skip_criteria(detection_params.skip_name);

detection_param_name = regexp(detection_result_txt,'\/?([-\w\.]+)\.txt','tokens');
detection_name = detection_param_name{1}{1};

fileID = fopen(detection_result_txt,'r');
[~, detection_result] = dwot_save_load_delegate(fileID, detection_params.save_mode);
fclose(fileID);

%                       detection_result(det_idx, end),... % detection score
%                       detection_result(det_idx, 1:4),... % bbox
%                       detection_result(det_idx, 11)));  % templateIdx
% detection.file_name    = detection_result.file_names;
% detection.bbox         = detection_result.prediction_boxes;
% detection.score        = detection_result.prediction_scores;
% detection.detector_idx = double(detection_result.prediction_template_indexes);
% detection.cnn_score    = double(detection_result.proposal_scores);

[unique_files, ~, unique_file_idx] = unique(detection_result.file_names);
n_unique_files = numel(unique_files);

[gtids,t] = textread(sprintf(VOCopts.imgsetpath,...
                  [detection_params.LOWER_CASE_CLASS '_' detection_params.TYPE]),'%s %d');

N_IMAGE = length(gtids);

tp_struct(n_unique_files)   = struct('gtBB',[],'prdBB',[],'diff',[],'truncated',[],'score',[],...
                                     'im',[],'detector_id',[]);
fp_struct(n_unique_files)   = struct('BB',[],'score',[],'im',[],'detector_id',[]);
fn_struct(n_unique_files)   = struct('BB',[],'diff',[],'truncated',[],'im',[]);

detector_struct = {};

npos = 0;
tp = cell(1,N_IMAGE);
fp = cell(1,N_IMAGE);
detScore = cell(1,N_IMAGE);

if b_compute_view
    tp_view = cell(1,N_IMAGE);
    fp_view = cell(1,N_IMAGE);
end

image_count = 1;
gt_orig =struct('BB',[],'diff',[],'det',[],'truncated',[],'occluded',[]);

for unique_image_idx=1:n_unique_files
    recs = PASreadrecord(sprintf(VOCopts.annopath,unique_files{unique_image_idx}));
    [~, file_name, ~] = fileparts(recs.filename);
    % curr_file_idx = cellfun(@(x) ~isempty(strmatch(x,file_name)), detection.file_name);
    curr_file_idx = find(unique_file_idx == unique_image_idx);
    if nnz(curr_file_idx) == 0 
      continue;
    end
    if mod(unique_image_idx,50) == 0; fprintf('.'); end;
  
    % read annotation
    clsinds = strmatch(detection_params.LOWER_CASE_CLASS,{recs.objects(:).class},'exact');
  
    [b_skip, ground_truth_obj_idx ] =dwot_skip_criteria(recs.objects(clsinds), skip_criteria);
    if b_skip; continue; end
  
    im = imread([VOCopts.datadir, recs.imgname]);
    imSz = size(im);
    
    ground_truth_bounding_boxes = cat(1, recs.objects(clsinds).bbox);
    difficult_object_indexes = [recs.objects(clsinds).difficult];
    truncated_object_indexes = [recs.objects(clsinds).truncated];

    % gt_orig.BB = cat(1, recs.objects(clsinds).bbox)';
    % gt_orig.diff = [recs.objects(clsinds).difficult];
    % gt_orig.truncated = [recs.objects(clsinds).truncated];
    % gt_orig.det = zeros(length(clsinds),1);
   
    % PASCAL 12 has additional annotation
    if strcmp(detection_params.DATA_SET, 'PASCAL12')
      occluded_object_indexes = [recs.objects(clsinds).occluded];
    end
    
    prediction_score = detection_result.prediction_scores(curr_file_idx);
    valid_prediction_idx = find(prediction_score > -inf);
    
    if ~isempty(valid_prediction_idx)
        curr_file_idx = curr_file_idx(valid_prediction_idx);
%         [~, unique_cnn_idx] = unique(detection.score(curr_file_idx));
%         curr_file_idx = curr_file_idx(unique_cnn_idx);

        % only bounding box
        prediction_bounding_box = detection_result.prediction_boxes(curr_file_idx,:)/image_scale_factor;

        % formatted such that it can be used in rendering
        formatted_bounding_box = [ prediction_bounding_box zeros(nnz(curr_file_idx),6),...
            double(detection_result.prediction_template_indexes(curr_file_idx,:)),...
            detection_result.prediction_scores(curr_file_idx)];

        % non maximal suppression on the bounding boxes
        [formatted_bounding_box, bounding_box_idx] = esvm_nms(formatted_bounding_box, nms_threshold);

        % non maximal suppressed prediction boxes
        prediction_bounding_box = prediction_bounding_box(bounding_box_idx,:);

        % clip the prediction 
        prediction_bounding_box_clip = clip_to_image(prediction_bounding_box, [1 1 imSz(2) imSz(1)]);

        n_prediction   = size(prediction_bounding_box,1);
        n_ground_truth = size(ground_truth_bounding_boxes,1);

        % evaluate prediction using the clipped prediction
        [tp{unique_image_idx}, fp{unique_image_idx}, prediction_iou, gt_idx_of_prediction] = ...
                dwot_evaluate_prediction(prediction_bounding_box_clip,...
                            ground_truth_bounding_boxes, param.min_overlap, ~ground_truth_obj_idx);
        detScore{unique_image_idx} = formatted_bounding_box(:,end)';

        if 1
            formatted_bounding_box(:,9) = prediction_iou;
            
            subplot(221);
            imagesc(im); axis equal; axis off; axis tight;
            
            subplot(222);
            imagesc(im);
            for draw_gt_idx = find(ground_truth_obj_idx)
                box_text = sprintf('id:%d', draw_gt_idx);
                current_draw_gt_bbox = ground_truth_bounding_boxes(draw_gt_idx,:);
                rectangle('position', dwot_bbox_xy_to_wh(current_draw_gt_bbox),'edgecolor',[0.5 0.5 0.5],'LineWidth',3);
                text(current_draw_gt_bbox(1) + 1 , current_draw_gt_bbox(2), box_text, 'BackgroundColor','w','EdgeColor',[0.5 0.5 0.5],'VerticalAlignment','bottom');
            end
            axis equal; axis off; axis tight;
            
            subplot(223);
            imagesc(im);
            for draw_pred_idx = find(tp{unique_image_idx})
                box_text = sprintf('s:%0.2f ov:%0.2f', formatted_bounding_box(draw_pred_idx, end), formatted_bounding_box(draw_pred_idx, 9));
                current_draw_gt_bbox = prediction_bounding_box(draw_pred_idx,:);
                rectangle('position', dwot_bbox_xy_to_wh(current_draw_gt_bbox),'edgecolor',[0.5 0.5 0.5],'LineWidth',3);
                text(current_draw_gt_bbox(1) + 1 , current_draw_gt_bbox(2), box_text, 'BackgroundColor','w','EdgeColor',[0.5 0.5 0.5],'VerticalAlignment','bottom');
            end
            axis equal; axis off; axis tight;
            
            subplot(224);
            imagesc(im);
            for draw_pred_idx = find(fp{unique_image_idx})
                box_text = sprintf('s:%0.2f ov:%0.2f', formatted_bounding_box(draw_pred_idx, end), formatted_bounding_box(draw_pred_idx, 9));
                current_draw_gt_bbox = prediction_bounding_box(draw_pred_idx,:);
                rectangle('position', dwot_bbox_xy_to_wh(current_draw_gt_bbox),'edgecolor',[0.5 0.5 0.5],'LineWidth',3);
                text(current_draw_gt_bbox(1) + 1 , current_draw_gt_bbox(2), box_text, 'BackgroundColor','w','EdgeColor',[0.5 0.5 0.5],'VerticalAlignment','bottom');
            end
            axis equal; axis off; axis tight;
            
%             subplot(223);
%             dwot_draw_overlap_rendering(im, formatted_bounding_box(tp{unique_image_idx}, :),...
%                         detectors, 5, 50, true, [0.2, 0.8, 0], param.color_range, 1 );
% 
%             subplot(224);
%             dwot_draw_overlap_rendering(im, formatted_bounding_box(fp{unique_image_idx}, :),...
%                         detectors, 5, 50, true, [0.2, 0.8, 0], [-inf 50:10:100 inf], 1 );
            
            drawnow
            waitforbuttonpress;
        end
        
        % Read viewpoint annotation
        if b_compute_view && ~isempty(clsinds)
            filename = fullfile(param.PASCAL3D_ANNOTATION_PATH,...
                                sprintf('%s_pascal/%s.mat', detection_params.LOWER_CASE_CLASS, file_name));
            view_annotation = load(filename);
            object_annotations = view_annotation.record.objects;
            object_annotations = object_annotations(clsinds); % Select only the object with the same class

            ground_truth_bounding_boxes = reshape([object_annotations.bbox],4,[])';

            % to evaluate also using viewpoint gather viewpoint info
            prediction_azimuth = cellfun(@(x) x.az, detectors(formatted_bounding_box(:,11)));
            ground_truth_azimuth = cellfun(@(x) x.azimuth, {object_annotations.viewpoint});

            [tp_view{unique_image_idx}, fp_view{unique_image_idx}, prediction_view_iou, gt_idx_of_view_prediction] =...
                    dwot_evaluate_prediction(prediction_bounding_box_clip, ground_truth_bounding_boxes,...
                            param.min_overlap, ~ground_truth_obj_idx,...
                            prediction_azimuth, ground_truth_azimuth, max_azimuth_difference,...
                            prediction_azimuth_rotation_direction, prediction_azimuth_offset);

            % Confusion Matrix
            [confusion_statistics] = dwot_gather_confusion_statistics(confusion_statistics,...
                    ground_truth_azimuth, prediction_azimuth, gt_idx_of_prediction,...
                    n_views, prediction_azimuth_rotation_direction, prediction_azimuth_offset);

            if 0
                tp_logical = tp_view{image_idx};
                formatted_bounding_box(:,9) = prediction_view_iou;
                formatted_bounding_box(:,10) = mod(prediction_azimuth_rotation_direction * prediction_azimuth +...
                                                    prediction_azimuth_offset,360)';
                
                dwot_visualize_result_with_azimuth(im, formatted_bounding_box, tp_logical,...
                                      ground_truth_bounding_box, ground_truth_azimuth, detectors, param.color_range);
                
                save_name = sprintf('%s_img_%s.jpg', detection_name, file_name);
                print('-djpeg','-r150',fullfile(save_path, save_name));
            end
        else
            tp_view{unique_image_idx} = false(1, n_prediction);
            fp_view{unique_image_idx} = true(1,  n_prediction);
        end

        %% Collect statistics
        % Per Detector Statistics

      %   for bbs_idx = 1:size(bbsNMS,1)
      %     detector_idx = bbsNMS(bbs_idx, 11);
      %     if numel(detector_struct) < detector_idx || isempty(detector_struct{detector_idx})
      %       detector_struct{detector_idx} = {};
      %     end
      %     detector_struct{detector_idx}{numel(detector_struct{detector_idx}) + 1} = struct('BB',bbsNMS(bbs_idx,:),'im',recs.imgname);
      %   end

        % TP collection
        correct_prediction_idx = find(gt_idx_of_prediction > 0);
        gt_idx = gt_idx_of_prediction(correct_prediction_idx);
        if ~isempty(gt_idx)
            tp_struct(image_count).gtBB = ground_truth_bounding_boxes(gt_idx, :);
            tp_struct(image_count).predBB = formatted_bounding_box(correct_prediction_idx ,1:4)';
            tp_struct(image_count).diff = difficult_object_indexes;
            tp_struct(image_count).truncated = truncated_object_indexes;
            if strcmp(detection_params.DATA_SET, 'PASCAL12')
                tp_struct(image_count).occluded = occluded_object_indexes;
            end
            tp_struct(image_count).score = formatted_bounding_box(correct_prediction_idx, end);
            tp_struct(image_count).im = repmat({recs.imgname}, numel(gt_idx), 1);
            tp_struct(image_count).detector_id = formatted_bounding_box(correct_prediction_idx, 11);
        end

        % FP collection
        bbsIdx = find(fp{unique_image_idx});

        if ~isempty(bbsIdx)
            fp_struct(image_count).BB = formatted_bounding_box(bbsIdx,1:4)';
            fp_struct(image_count).score = formatted_bounding_box(bbsIdx,end);
            fp_struct(image_count).im = repmat({recs.imgname}, numel(bbsIdx),1);
            fp_struct(image_count).detector_id = formatted_bounding_box(bbsIdx,11);
        end

        % FN collection
        gt_idx = find(gt_orig.det == 0);

        if ~isempty(gt_idx)
            fn_struct(image_count).BB = ground_truth_bounding_boxes(:, gt_idx);
            fn_struct(image_count).diff = difficult_object_indexes;
            fn_struct(image_count).truncated = truncated_object_indexes;
            % Only PASCAL 11,12 have the label
            if strcmp(detection_params.DATA_SET, 'PASCAL12')
                fn_struct(image_count).occluded = occluded_object_indexes;
            end
            fn_struct(image_count).im = repmat({recs.imgname}, numel(gt_idx), 1);
        end
    
    end
    image_count = image_count + 1;
    
    % npos=npos+sum(~gt_orig.diff);
    npos = npos + nnz(ground_truth_obj_idx);
end


detScore = cell2mat(detScore);
fp = cell2mat(cellfun(@(x) double(x), fp, 'UniformOutput',false));
tp = cell2mat(cellfun(@(x) double(x), tp, 'UniformOutput',false));

[~, si] =sort(detScore,'descend');
fpSort = cumsum(fp(si));
tpSort = cumsum(tp(si));

recall = tpSort/npos;
precision = tpSort./(fpSort + tpSort);

ap = VOCap(recall', precision');

close all

if b_compute_view
    fp_view = cell2mat(cellfun(@(x) double(x), fp_view, 'UniformOutput',false));
    tp_view = cell2mat(cellfun(@(x) double(x), tp_view, 'UniformOutput',false));
    fpSort_view = cumsum(fp_view(si));
    tpSort_view = cumsum(tp_view(si));

    recall_view = tpSort_view/npos;
    precision_view = tpSort_view./(fpSort_view + tpSort_view);

    avp = VOCap(recall_view', precision_view');
    
    fprintf('\nAP = %.4f AA = %.4f\n', ap, avp);
    
    subplot(121);
    plot(recall, precision, 'r', 'LineWidth',3);
    hold on;
    plot(recall_view, precision_view, 'g', 'LineWidth',3);

    xlabel('Recall');
    ti = sprintf('Average Precision = %.3f Average Viewpoint Precision = %.3f', 100*ap, 100*avp);
    title(ti);
    axis equal; axis tight;
    axis([0 1 0 1]);

    confusion_precision = bsxfun(@rdivide, confusion_statistics, sum(confusion_statistics));
    mppe = mean(diag(confusion_precision));

    subplot(122);
    colormap cool;
    imagesc(confusion_precision); colormap; colorbar; axis equal; axis tight;
    ti = sprintf('Viewpoint confusion matrix MPPE=%0.2f', 100*mppe);
    title(ti);
    xlabel('ground truth viewpoint index');
    ylabel('prediction viewpoint index');

    set(gcf,'color','w');
    drawnow;
else
    fprintf('\nAP = %.4f\n', ap);
    
    plot(recall, precision, 'r', 'LineWidth',3);

    xlabel('Recall');
    ti = sprintf('Average Precision = %.3f', 100*ap);
    title(ti);
    axis equal; axis tight;
    axis([0 1 0 1]);
    
    set(gcf,'color','w');
    drawnow;
end

return;

% Sort the scores for each detectors and print  
if visualize
    if ~exist(save_path,'dir')
      mkdir(save_path);
    end

    if 0
        for detector_idx = 1:numel(detector_struct)
          if isempty(detector_struct{detector_idx})
            continue;
          end

          score = cellfun(@(x) x.BB(end), detector_struct{detector_idx});
          [~, sorted_idx] = sort(score,'descend');

          count = 1;
          for idx = sorted_idx(1:min(7,numel(sorted_idx)))
            curr_detection = detector_struct{detector_idx}{idx};

            im_name = curr_detection.im;
            im = imread([VOCopts.datadir,im_name]);

            BB = curr_detection.BB(1:4);

            subplot(221);
            imagesc(im);
            rectangle('position', BB - [0 0 BB(1:2)],'edgecolor','b','linewidth',2);
            axis equal; axis tight;

            % Rendering
            subplot(223);
            imagesc(renderings{detector_idx});
            axis equal; axis tight;

            % Overlay
            subplot(224);
            dwot_draw_overlap_detection(im,  curr_detection.BB, renderings, 1, 50, true, [0.5, 0.5, 0], color_range);
            axis equal; axis tight;

            drawnow;
            spaceplots();

            save_name = sprintf('%s_detector_%d_sort_%04d.jpg', detection_name, detector_idx, count);
            count = count + 1;
            print('-djpeg','-r150',fullfile(save_path, save_name));
          end
        end
    end

    % sort the TP and FP according to the score and and print them
    % Concatenate all arrays, we used structure to speed up memory allocation

    non_empty_idx = cellfun(@(x) ~isempty(x), {tp_struct.diff});

    tp.gtBB       = cell2mat({tp_struct.gtBB});
    tp.predBB     = cell2mat({tp_struct.predBB});
    tp.diff       = cell2mat({tp_struct(non_empty_idx).diff});
    tp.truncated  = cell2mat({tp_struct(non_empty_idx).truncated});
    tp.occluded  = cell2mat({tp_struct(non_empty_idx).occluded});
    tp.score      = cell2mat({tp_struct(non_empty_idx).score}');
    tp.im         = convert_doubly_deep_cell({tp_struct(non_empty_idx).im});
    tp.detector_id = cell2mat({tp_struct.detector_id}');

    [~, sorted_idx] = sort(tp.score,'descend');

    count = 1;
    for idx = sorted_idx'
      im_cell = tp.im(idx);
      im = imread([VOCopts.datadir,im_cell{1}]);
      % im = imresize(im, image_scale_factor);

      gtBB = tp.gtBB(:,idx);
      predBB = tp.predBB(:,idx);

      subplot(121);
      imagesc(im);
      axis equal; axis tight;

      box_position = gtBB(1:4)' + [0 0 -gtBB(1:2)'];
    
      % if detector id available (positive number), print it
      [~, color_idx] = histc(tp.score(idx), color_range);
      curr_color = color_map(color_idx, :);
      rectangle('position', box_position,'edgecolor',[0.5 0.5 0.5],'LineWidth',3);
      rectangle('position', box_position,'edgecolor',curr_color,'LineWidth',1);
      text(box_position(1) + 1 , box_position(2), sprintf(' s:%0.2f',tp.score(idx)),...
          'BackgroundColor', curr_color,'EdgeColor',[0.5 0.5 0.5],'VerticalAlignment','bottom');
      

    
      % Crop
      subplot(122);
      imagesc(im(gtBB(2):gtBB(4), gtBB(1):gtBB(3), :));
      axis equal; axis tight;

      drawnow;
      spaceplots();

      save_name = sprintf('%s_tp_sort_%04d.jpg', detection_name, count);
      count = count + 1;
      print('-djpeg','-r150',fullfile(save_path, save_name));
      % waitforbuttonpress;
    end


    % False Positive
    non_empty_idx = cellfun(@(x) ~isempty(x), {fp_struct.score});

    fp.BB       = cell2mat({fp_struct.BB});
    fp.score      = cell2mat({fp_struct(non_empty_idx).score}');
    fp.im         = convert_doubly_deep_cell({fp_struct(non_empty_idx).im});
    fp.detector_id = cell2mat({fp_struct.detector_id}');

    [~, sorted_idx] = sort(fp.score,'descend');
    count = 1;

    for idx = sorted_idx(1:600)'
      im_cell = fp.im(idx);
      im = imread([VOCopts.datadir,im_cell{1}]);
      % im = imresize(im, image_scale_factor);
      imSz = size(im);

      BB = clip_to_image( round(fp.BB(:,idx))', [1 1 imSz(2) imSz(1)]) ;

      subplot(121);
      imagesc(im);
      rectangle('position', BB - [0 0 BB(1:2)],'edgecolor','r','linewidth',2);
      axis equal; axis tight;

      [~, color_idx] = histc(fp.score(idx), color_range);
      curr_color = color_map(color_idx, :);
      rectangle('position', BB,'edgecolor',[0.5 0.5 0.5],'LineWidth',3);
      rectangle('position', BB,'edgecolor',curr_color,'LineWidth',1);
      text(BB(1) + 1 , BB(2), sprintf(' s:%0.2f',fp.score(idx)),...
          'BackgroundColor', curr_color,'EdgeColor',[0.5 0.5 0.5],'VerticalAlignment','bottom');
      
      
      % Crop
      subplot(122);
      imagesc(im(BB(2):BB(4), BB(1):BB(3), :));
      axis equal; axis tight;



      drawnow;
      spaceplots();

      save_name = sprintf('%s_fp_sort_%04d.jpg', detection_name, count);
      count = count + 1;
      print('-djpeg','-r150',fullfile(save_path, save_name));
    end


    % False Negative
    non_empty_idx = cellfun(@(x) ~isempty(x), {fn_struct.diff});

    fn.BB       = cell2mat({fn_struct.BB});
    fn.diff       = cell2mat({fn_struct(non_empty_idx).diff});
    fn.truncated  = cell2mat({fn_struct(non_empty_idx).truncated});
    fn.im         = convert_doubly_deep_cell({fn_struct(non_empty_idx).im});
    count = 1;

    close all;
    for idx = 1:numel(fn.diff)
      im_cell = fn.im(idx);
      im = imread([VOCopts.datadir,im_cell{1}]);
%       im = imresize(im, image_scale_factor);

      BB = fn.BB(:,idx) ;
      plot_title = '';
      if fn.diff(idx)
        continue;
        plot_title = [plot_title ' difficult'];
      end

      if fn.truncated(idx)
        plot_title = [plot_title ' truncated'];
      end
      
%       if fn.occluded(idx)
%         plot_title = [plot_title ' truncated'];
%       end
      subplot(121);
      imagesc(im);
      rectangle('position', BB' - [0 0 BB(1:2)'],'edgecolor','b','linewidth',2);
      axis equal; axis tight;

      title(plot_title);

      % Crop
      subplot(122);
      imagesc(im(BB(2):BB(4), BB(1):BB(3), :));
      axis equal; axis tight;

      drawnow;
      spaceplots();

      save_name = sprintf('%s_fn_sort_%04d.jpg', detection_name, count);
      count = count + 1;
      print('-djpeg','-r150',fullfile(save_path, save_name));
    end
end




function plain_cell = convert_doubly_deep_cell(doubly_cell)

count = 1;
plain_cell = {};
for cell_idx = 1:numel(doubly_cell)
  for doubly_cell_idx = 1:numel(doubly_cell{cell_idx})
    plain_cell{count} = doubly_cell{cell_idx}{doubly_cell_idx};
    count = count + 1;
  end
end
