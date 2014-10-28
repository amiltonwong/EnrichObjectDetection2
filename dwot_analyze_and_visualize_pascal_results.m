function [ ap ] = dwot_analyze_and_visualize_pascal_results( detection_result_txt, ...
    detectors, save_path, VOCopts, param, skip_criteria, color_range, nms_threshold, visualize)

if ~exist('nms_threshold','var') || isempty(nms_threshold)
    if ~isfield(param, 'nms_threshold')
      nms_threshold = 0.4;
    else
      nms_threshold = param.nms_threshold;
    end
end

if ~exist('skip_criteria','var') || isempty(skip_criteria)
  % skip_criteria = {'empty', 'difficult','truncated'};
  skip_criteria = {'empty'};
end

if ~exist('color_range','var')
  color_range = [-inf 100:20:300 inf];
end

renderings = cellfun(@(x) x.rendering_image, detectors, 'UniformOutput', false);

% PASCAL_car_val_init_0_Car_each_7_lim_300_lam_0.0150_a_24_e_2_y_1_f_2_scale_2.00_sbin_6_level_10_nms_0.40_skp_e
% Group inside group not supported in matlab
detection_params = dwot_detection_params_from_name(detection_result_txt);

% SNames = fieldnames(temp); 
% for loopIndex = 1:numel(SNames) 
%     stuff = S.(SNames{loopIndex})
% end

detection_param_name = regexp(detection_result_txt,'\/?([\w\.]+)\.txt','tokens');
detection_name = detection_param_name{1}{1};

[gtids,t] = textread(sprintf(VOCopts.imgsetpath,[detection_params.LOWER_CASE_CLASS '_' detection_params.TYPE]),'%s %d');

% Find image scaling factor
temp = regexp(detection_result_txt,'\d_scale_([\d+.]+)','tokens');
image_scale_factor = str2double(temp{1});


% PASCAL_car_val_init_0_Car_each_7_lim_300_lam_0.0150_a_24_e_2_y_1_f_2_scale_2.00_sbin_6_level_15_nms_0.40_skp_e_server_102.txt
    
N_IMAGE = length(gtids);

gt(length(gtids))=struct('BB',[],'diff',[],'det',[]);

fileID = fopen(detection_result_txt,'r');
detection_result = textscan(fileID,'%s %f %f %f %f %f %d');
fclose(fileID);
%                       detection_result(det_idx, end),... % detection score
%                       detection_result(det_idx, 1:4),... % bbox
%                       detection_result(det_idx, 11)));  % templateIdx

% Sort files to corresponding filed
detection.file_name = detection_result{1};
detection.bbox = cell2mat(detection_result(3:6));
detection.score = detection_result{2};
detection.detector_idx = double(detection_result{7});

[unique_files, ~, unique_file_idx] = unique(detection.file_name);
n_unique_files = numel(unique_files);

tp_struct   = struct('gtBB',[],'prdBB',[],'diff',[],'truncated',[],'score',[],'im',[],'detector_id',[]);
fp_struct   = struct('BB',[],'score',[],'im',[],'detector_id',[]);
fn_struct   = struct('BB',[],'diff',[],'truncated',[],'im',[]);

detector_struct = {};

npos = 0;
tp = cell(1,N_IMAGE);
fp = cell(1,N_IMAGE);
detScore = cell(1,N_IMAGE);

image_count = 1;

for imgIdx=1:n_unique_files
  recs=PASreadrecord(sprintf(VOCopts.annopath,unique_files{imgIdx}));
  [~, file_name, ~] = fileparts(recs.filename);
  % curr_file_idx = cellfun(@(x) ~isempty(strmatch(x,file_name)), detection.file_name);
  curr_file_idx = find(unique_file_idx == imgIdx);
  if nnz(curr_file_idx) == 0 
    continue;
  end
  if mod(imgIdx,10) == 0; fprintf('.'); end;

  % read annotation
  clsinds = strmatch(detection_params.LOWER_CASE_CLASS,{recs.objects(:).class},'exact');

  [b_skip_img, gt_to_use] = dwot_skip_criteria(recs.objects(clsinds), skip_criteria); 
  
  if b_skip_img
      continue;
  end
gt_to_use

%   im = imread([VOCopts.datadir, recs.imgname]);
%   im = imresize(im, image_scale_factor);
%   imSz = size(im);
  
  gt(imgIdx).BB = cat(1, recs.objects(clsinds).bbox)';
  gt(imgIdx).diff = [recs.objects(clsinds).difficult];
  gt(imgIdx).truncated = [recs.objects(clsinds).truncated];
  gt(imgIdx).det = zeros(length(clsinds),1);
  

  bbs = [ detection.bbox(curr_file_idx,:)/param.image_scale_factor zeros(nnz(curr_file_idx),6) detection.detector_idx(curr_file_idx,:) detection.score(curr_file_idx)];
  bbsNMS = esvm_nms(bbs, nms_threshold);
  
  bbsNMS_clip = bbsNMS;
  % bbsNMS_clip = clip_to_image(bbsNMS, [1 1 imSz(2) imSz(1)]);
  [bbsNMS_clip, tp{imgIdx}, fp{imgIdx}, detScore{imgIdx}, gt(imgIdx)] = ...
                    dwot_compute_positives(bbsNMS_clip, gt(imgIdx), param);
  bbsNMS(:,9) = bbsNMS_clip(:,9);
  
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
  gtIdx = find(gt(imgIdx).det > 0);
  bbsIdx = gt(imgIdx).det(gtIdx);
  if ~isempty(gtIdx)
    tp_struct(image_count).gtBB = gt(imgIdx).BB(:, gtIdx);
    tp_struct(image_count).predBB = bbsNMS(bbsIdx,1:4)';
    tp_struct(image_count).diff = logical(gt(imgIdx).diff(gtIdx));
    tp_struct(image_count).truncated = logical(gt(imgIdx).truncated(gtIdx));
    tp_struct(image_count).score = bbsNMS_clip(bbsIdx,end);
    tp_struct(image_count).im = repmat({recs.imgname}, numel(gtIdx), 1);
    tp_struct(image_count).detector_id = bbsNMS_clip(bbsIdx,11);
  end
  
  % FP collection
  bbsIdx = find(fp{imgIdx});
  
  if ~isempty(bbsIdx)
    fp_struct(image_count).BB = bbsNMS_clip(bbsIdx,1:4)';
    fp_struct(image_count).score = bbsNMS_clip(bbsIdx,end);
    fp_struct(image_count).im = repmat({recs.imgname}, numel(bbsIdx),1);
    fp_struct(image_count).detector_id = bbsNMS_clip(bbsIdx,11);
  end
  
  % FN collection
  gtIdx = find(gt(imgIdx).det == 0);
  
  if ~isempty(gtIdx)
    fn_struct(image_count).BB = gt(imgIdx).BB(:, gtIdx);
    fn_struct(image_count).diff = logical(gt(imgIdx).diff(gtIdx));
    fn_struct(image_count).truncated = logical(gt(imgIdx).truncated(gtIdx));
    fn_struct(image_count).im = repmat({recs.imgname}, numel(gtIdx), 1);
  end
  
  image_count = image_count + 1;
  
  npos=npos+sum(~gt(imgIdx).diff);
  
end


detScore = cell2mat(detScore);
fp = cell2mat(fp);
tp = cell2mat(tp);

[~, si] =sort(detScore,'descend');
fpSort = cumsum(fp(si));
tpSort = cumsum(tp(si));

recall = tpSort/npos;
precision = tpSort./(fpSort + tpSort);

ap = VOCap(recall', precision');
fprintf('\nAP = %.4f\n', ap);

plot(recall, precision, 'r', 'LineWidth',3);
xlabel('Recall');

tit = sprintf('Average Precision = %.3f', 100*ap);
title(tit);
axis([0 1 0 1]);
set(gcf,'color','w');
drawnow;
    
    
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
            im = imresize(im, image_scale_factor);

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
    tp.score      = cell2mat({tp_struct(non_empty_idx).score}');
    tp.im         = convert_doubly_deep_cell({tp_struct(non_empty_idx).im});
    tp.detector_id = cell2mat({tp_struct.detector_id}');

    [~, sorted_idx] = sort(tp.score,'descend');

    count = 1;
    for idx = sorted_idx'
      im_cell = tp.im(idx);
      im = imread([VOCopts.datadir,im_cell{1}]);
%       im = imresize(im, image_scale_factor);

      gtBB = tp.gtBB(:,idx);
      predBB = tp.predBB(:,idx);

      subplot(221);
      imagesc(im);
      rectangle('position', gtBB' - [0 0 gtBB(1:2)'],'edgecolor','b','linewidth',2);
      rectangle('position', predBB' - [0 0 predBB(1:2)'],'edgecolor','r','linewidth',2);
      axis equal; axis tight;

      % Crop

      subplot(222);
      imagesc(im(gtBB(2):gtBB(4), gtBB(1):gtBB(3), :));
      axis equal; axis tight;

      % Rendering
      subplot(223);
      imagesc(renderings{tp.detector_id(idx)});
      axis equal; axis tight;

      % Overlay
      subplot(224);
      dwot_draw_overlap_detection(im, [predBB' zeros(1,6) tp.detector_id(idx) tp.score(idx)], renderings, 1, 50, true, [0.5, 0.5, 0], param.color_range );
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
%       im = imresize(im, image_scale_factor);
      imSz = size(im);

      BB = clip_to_image( round(fp.BB(:,idx))', [1 1 imSz(2) imSz(1)]) ;

      subplot(221);
      imagesc(im);
      rectangle('position', BB - [0 0 BB(1:2)],'edgecolor','r','linewidth',2);
      axis equal; axis tight;

      % Crop
      subplot(222);
      imagesc(im(BB(2):BB(4), BB(1):BB(3), :));
      axis equal; axis tight;

      % Rendering
      subplot(223);
      imagesc(renderings{fp.detector_id(idx)});
      axis equal; axis tight;

      % Overlay
      subplot(224);
      dwot_draw_overlap_detection(im, [BB zeros(1,6) fp.detector_id(idx) fp.score(idx)], renderings, 1, 50, true, [0.5, 0.5, 0], param.color_range);
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
