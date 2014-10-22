addpath('HoG');
addpath('HoG/features');
addpath('Util');
addpath('DecorrelateFeature/');
addpath('../MatlabRenderer/');
addpath('../MatlabRenderer/bin');
addpath('../MatlabCUDAConv/');
addpath('3rdParty/SpacePlot');
addpath('3rdParty/MinMaxSelection/');

% Tests average model
% The average template is generated using multiple renderings
% 
% see dwot_avg_model.m
rng('default');
DATA_SET = '3DObject';
dwot_set_datapath;
% Computing Mode  = 0, CPU
%                 = 1, GPU
%                 = 2, Combined
COMPUTING_MODE = 1;
CLASS = 'Car';
SUB_CLASS = 'Sedan';
LOWER_CASE_CLASS = lower(CLASS);
TEST_TYPE = 'val';
SAVE_PATH = fullfile('Result',[LOWER_CASE_CLASS '_' TEST_TYPE]);
if ~exist(SAVE_PATH,'dir'); mkdir(SAVE_PATH); end

DEVICE_ID = 1; % 0-base indexing

if COMPUTING_MODE > 0
  gdevice = gpuDevice(DEVICE_ID + 1); % Matlab use 1 base indexing
  reset(gdevice);
  cos(gpuArray(1));
end

azs = 0:45:315;
els = 0:20:20;
fovs = [25 50];
yaws = 0;
n_cell_limit = [250];
lambda = [0.2];
detection_threshold = 80;

visualize_detection = true;
visualize_detector = false;

sbin = 4;
n_level = 15;
n_proposals = 5;

% Get all possible sub-classes
[ model_names, model_paths ] = dwot_get_cad_models('Mesh', CLASS, SUB_CLASS, {'3ds'});

honda_idx = ismember(model_names,'Honda-Accord-3');
model_names = model_names(honda_idx);
model_paths = model_paths(honda_idx);
%%%%%%%%%%%%%%% Set Parameters %%%%%%%%%%%%

dwot_get_default_params;

param.template_initialization_mode = 0; 
param.nms_threshold = 0.4;
param.model_paths = model_paths;

param.b_calibrate = 0;      % apply callibration if > 0
param.n_calibration_images = 100; 
% Calibration mode == 'gaussian', fit gaussian
%                  == 'linear' , put 0.01% of data to be above 1.
param.calibration_mode = 'none';

switch param.calibration_mode
  case 'gaussian'
    param.color_range = [-inf 4:0.5:10 inf];
    param.detection_threshold = 4;
    param.b_calibrate = 1;
  case 'linear'
    param.color_range = [-inf -0.2:0.1:3 inf];
    param.detection_threshold = -0.2;
    param.b_calibrate = 1;
  case 'none'
    param.color_range = [-inf 30:5:100 inf];
    param.detection_threshold = 30;
    param.b_calibrate = 0;
end

param.image_scale_factor = 1; % scale image accordingly and detect on the scaled image

% Tuning mode == 0, no tuning
%             == 1, MCMC
%             == 2, Breadth first search
%             == 3, Quasi-Newton method (BFGS)
param.proposal_tuning_mode = 1;


% Detection mode == 'dwot' ours
%                == 'cnn'
%                == 'dpm'
param.detection_mode = 'dwot';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% For Debuggin purpose only
% param.detectors              = detectors;
% param.detect_pyramid_padding = 10;
% param.min_overlap = 0.5;


% detector name
[ detector_model_name ] = dwot_get_detector_name(CLASS, SUB_CLASS, model_names, param);

detector_name = sprintf('%s_%s_lim_%d_lam_%0.3f_a_%d_e_%d_y_%d_f_%d',...
        LOWER_CASE_CLASS, detector_model_name, n_cell_limit, lambda,...
        numel(azs), numel(els), numel(yaws), numel(fovs));

detector_file_name = sprintf('%s.mat', detector_name);


%% Make Renderer
if ~exist('renderer','var') || (~exist(detector_file_name,'file')  && param.proposal_tuning_mode > 0)
    if exist('renderer','var')
        renderer.delete();
        clear renderer;
    end

    % Initialize renderer
    renderer = Renderer();
    if ~renderer.initialize(model_paths, 700, 700, 0, 0, 0, 0, 25)
      error('fail to load model');
    end
end

%% Make Detectors
if exist(detector_file_name,'file')
    load(detector_file_name);
else
    [detectors] = dwot_make_detectors_grid(renderer, azs, els, yaws, fovs, 1:length(model_names),...
        LOWER_CASE_CLASS, param, visualize_detector);
    [detectors, detector_table]= dwot_make_table_from_detectors(detectors);
    if sum(cellfun(@(x) isempty(x), detectors))
      error('Detector Not Completed');
    end
    eval(sprintf(['save -v7.3 ' detector_file_name ' detectors detector_table']));
    % detectors = dwot_make_detectors(renderer, azs, els, yaws, fovs, param, visualize_detector);
    % eval(sprintf(['save ' detector_name ' detectors']));
end


%% Calibrate detector. see param.b_calibrate
% calibration mode == 'gaussian'
%     tries to normalize the negative detection by normalizing it to follow standard distribution.
%     performs worse than without calibration
% calibration_mode == 'linear'
%     follow 'Seeing 3D chair' CVPR 14, calibration stage. Performs worse
if param.b_calibrate

    %%%%%%%%%% Since we do not have VOCopt for 3DObject, embed the ugly
    % loading part
    if isempty(server_id) || ~isempty(strmatch(server_id.num,'capri7'))
        VOC_PATH = '/home/chrischoy/Dataset/VOCdevkit/';
    else
        VOC_PATH = '/scratch/chrischoy/Dataset/VOCdevkit/';
    end
    if ismac
        VOC_PATH = '~/dataset/VOCdevkit/';
    end
    addpath(VOC_PATH);
    addpath([VOC_PATH, 'VOCcode']);

    curDir = pwd;
    eval(['cd ' VOC_PATH]);
    VOCinit;
    eval(['cd ' curDir]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    detector_name = sprintf('%s_cal_%s',detector_name, param.calibration_mode);
    detector_file_name = sprintf('%s.mat', detector_name);
    if exist(detector_file_name,'file')
        load(detector_file_name);
    else
        detectors = dwot_calibrate_detectors(detectors, LOWER_CASE_CLASS, VOCopts, param);
        eval(sprintf(['save -v7.3 ' detector_file_name ' detectors']));
    end
    param.detectors = detectors;
end


%% Make empty detection save file
detection_result_file = sprintf(['%s_%s_%s_%s_lim_%d_lam_%0.3f_a_%d_e_%d_y_%d_f_%d_scale_',...
            '%0.2f_sbin_%d_level_%d_server_%s.txt'],...
            DATA_SET, LOWER_CASE_CLASS, TEST_TYPE, detector_model_name, n_cell_limit, lambda,...
            numel(azs), numel(els), numel(yaws), numel(fovs), param.image_scale_factor, sbin,...
            n_level, server_id.num);
        
% Check duplicate file name and return different name
detection_result_file = dwot_save_detection([], SAVE_PATH, detection_result_file, [], true);
detection_result_common_name = regexp(detection_result_file, '\/?(?<name>.+)\.txt','names');
detection_result_common_name = detection_result_common_name.name;

if param.proposal_tuning_mode > 0
    detection_tuning_result_file = sprintf(['%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_scale_',...
            '%0.2f_sbin_%d_level_%d_nms_%0.2f_server_%s_tuning.txt'],...
            DATA_SET, LOWER_CASE_CLASS, TEST_TYPE, detector_model_name, n_cell_limit, lambda,...
            numel(azs), numel(els), numel(yaws), numel(fovs), param.image_scale_factor, sbin,...
            n_level, param.nms_threshold, server_id.num);

    % Check duplicate file name and return different name
    detection_tuning_result_file = dwot_save_detection([], SAVE_PATH, detection_tuning_result_file, [], true);
    detection_tuning_result_common_name = regexp(detection_tuning_result_file, '\/?(?<name>.+)\.txt','names');
    detection_tuning_result_common_name = detection_tuning_result_common_name.name;
end

fprintf('\nThe result will be saved on %s\n', detection_result_file);


%%%%% For Debuggin purpose only
param.detectors              = detectors;
param.detect_pyramid_padding = 10;
%%%%%%%%%%%%

%% Make templates, these are just pointers to the templates in the detectors,
% The following code copies variables to GPU or make pointers to memory
% according to the computing mode.
%
% The GPU templates accelerates the computation time since it is already loaded
% on GPU.
if COMPUTING_MODE == 0
  % for CPU convolution, use fconvblas which handles template inversion
  templates_cpu = cellfun(@(x) single(x.whow), detectors,'UniformOutput',false);
elseif COMPUTING_MODE == 1
  % for GPU convolution, we use FFT based convolution. invert template
  templates_gpu = cellfun(@(x) gpuArray(single(x.whow(end:-1:1,end:-1:1,:))), detectors,'UniformOutput',false);
elseif COMPUTING_MODE == 2
  templates_gpu = cellfun(@(x) gpuArray(single(x.whow(end:-1:1,end:-1:1,:))), detectors,'UniformOutput',false);
  templates_cpu = cellfun(@(x) single(x.whow), detectors,'UniformOutput',false);
else
  error('Computing mode undefined');
end


% curDir = pwd;
% eval(['cd ' VOC_PATH]);
% VOCinit;
% eval(['cd ' curDir]);
% 
% read annotation
[gt, image_path] = dwot_3d_object_dataset(DATA_PATH, CLASS);

N_IMAGE = length(gt);
N_IMAGE = ceil(N_IMAGE/4); % Define it as the validation set

npos = 0;
npos_view = 0;
tp = cell(1,N_IMAGE);
fp = cell(1,N_IMAGE);
tp_view = cell(1,N_IMAGE);
fp_view = cell(1,N_IMAGE);
detScore = cell(1,N_IMAGE);
detScore_view = cell(1,N_IMAGE);
detectorId = cell(1,N_IMAGE);
detectorId_view = cell(1,N_IMAGE);


% After post-processing proposal regions
tp_prop = cell(1,N_IMAGE);
fp_prop = cell(1,N_IMAGE);

detScore_prop = cell(1,N_IMAGE);
detectorId_prop = cell(1,N_IMAGE);

tp_per_template = cell(1,numel(templates_gpu));
fp_per_template = cell(1,numel(templates_gpu));

n_views = 8;
confusion_statistics = zeros(n_views, n_views);

for imgIdx = 1:N_IMAGE
    fprintf('%d/%d ',imgIdx,N_IMAGE);
    imgTic = tic;

    im = imread([image_path{imgIdx}]);
    img_file_name = regexp(image_path{imgIdx}, '\/(car_\d+\/\w+)\.','tokens');
    img_file_name = img_file_name{1}{1};
    imSz = size(im);
    if COMPUTING_MODE == 0
      [bbsAllLevel, hog, scales] = dwot_detect( im, templates_cpu, param);
      % [hog_region_pyramid, im_region] = dwot_extract_region_conv(im, hog, scales, bbsNMS, param);
      % [bbsNMS_MCMC] = dwot_mcmc_proposal_region(im, hog, scale, hog_region_pyramid, param);
    elseif COMPUTING_MODE == 1
      % [bbsNMS ] = dwot_detect_gpu_and_cpu( im, templates, templates_cpu, param);
      [bbsAllLevel, hog, scales] = dwot_detect_gpu( im, templates_gpu, param);
    elseif COMPUTING_MODE == 2
      [bbsAllLevel, hog, scales] = dwot_detect_combined( im, templates_gpu, templates_cpu, param);
    else
      error('Computing Mode Undefined');
    end
    fprintf(' convolution time : %0.4f\n', toc(imgTic));
    
    % Automatically sort them according to the score and apply NMS
    % bbsNMS = esvm_nms(bbsAllLevel,0.5);
    % Per Viewpoint NMS
    bbsNMS_per_template_nms = dwot_per_view_nms(bbsAllLevel, param.nms_threshold);
    bbsNMS_per_template_nms_mat = cell2mat(bbsNMS_per_template_nms);
    bbsNMS_clip_per_template_mat = clip_to_image(bbsNMS_per_template_nms_mat, [1 1 imSz(2) imSz(1)]);
    
    
    % Save detection with minimal nms suppression. The saved result will be
    % used to analyze nms threshold.
    dwot_save_detection(esvm_nms(bbsAllLevel, 0.7), SAVE_PATH, detection_result_file, ...
                                            img_file_name, false, 1);
                       
    % Compute overlap of per view nms detections
    [bbsNMS_clip_per_template_mat, ~, ~, ~] = dwot_compute_positives_view(bbsNMS_clip_per_template_mat, gt{imgIdx}, detectors, param);
    if numel(bbsNMS_clip_per_template_mat) > 0
        bbsNMS_per_template_nms_mat(:,9) = bbsNMS_clip_per_template_mat(:,9); % copy overlap to original non-clipped detection
    end
    % Gather statistics
    % [tp_per_template, fp_per_template] = dwot_gather_viewpoint_statistics(tp_per_template, fp_per_template, bbsNMS_clip_per_template_mat, 0.75);
    
    % NMS again
    [bbsNMS_clip_per_template_nms_mat_nms, nms_idx] = esvm_nms(bbsNMS_clip_per_template_mat, param.nms_threshold);
    bbsNMS_per_template_nms_mat_nms = bbsNMS_per_template_nms_mat(nms_idx,:);
    
    % [bbsNMS_clip_per_template, tp_view{imgIdx}, fp_view{imgIdx}, ~] = dwot_compute_positives_view(bbsNMS_clip_per_template_mat, gt{imgIdx}, detectors, param);
    % Applying NMS dramatically reduce false positives
    [bbsNMS_clip_per_template_nms_mat_nms, tp{imgIdx}, fp{imgIdx}, detScore{imgIdx}, ~] = dwot_compute_positives(bbsNMS_clip_per_template_nms_mat_nms, gt{imgIdx}, param);

    % dwot_compute_positives_view will return GT index at the 10th column of bbsNMS
    [bbsNMS_clip_per_template_nms_mat_nms, tp_view{imgIdx}, fp_view{imgIdx}, detScore_view{imgIdx}, ~] = dwot_compute_positives_view(bbsNMS_clip_per_template_nms_mat_nms, gt{imgIdx}, detectors, param);
    
    % Confusion Matrix
    confusion_statistics = dwot_gather_confusion_statistics(confusion_statistics, detectors, gt{imgIdx}, bbsNMS_clip_per_template_nms_mat_nms, n_views);

    if visualize_detection
        tpIdx = logical(tp{imgIdx});
        
        % Original images
        subplot(221);
        imagesc(im); axis off; axis equal;

        % True positives
        subplot(222);
        % dwot_draw_overlap_detection(im, bbsNMS_per_template_nms_mat_nms(tpIdx,:), renderings, inf, 50, visualize_detection, [0.2, 0.8, 0] );
        im_temp = dwot_draw_overlap_rendering(im, bbsNMS_per_template_nms_mat_nms(tpIdx,:),...
                            detectors, inf, 50, false, [0.2, 0.8, 0], param.color_range, 1 );
        imagesc(im_temp); axis off; axis equal; axis tight;
        
        % True positives
        subplot(223);
        % dwot_draw_overlap_detection(im, bbsNMS_per_template_nms_mat_nms(tpIdx,:), renderings, inf, 50, visualize_detection, [0.2, 0.8, 0] );
        dwot_draw_overlap_rendering(im, bbsNMS_per_template_nms_mat_nms(tpIdx,:), detectors,...
                            inf, 50, visualize_detection, [0.2, 0.8, 0], param.color_range, 1  );
        
        % Tuning Result
        % subplot(223);
        
        % False positives
        subplot(224);
        % dwot_draw_overlap_detection(im, bbsNMS_per_template_nms_mat_nms(~tpIdx,:), renderings, n_proposals, 50, visualize_detection);
        dwot_draw_overlap_rendering(im, bbsNMS_per_template_nms_mat_nms(~tpIdx,:), detectors,...
                            5, 50, visualize_detection, [0.2, 0.8, 0], param.color_range, 1  );
        
        drawnow;
        spaceplots();
        save_name = sprintf('%s_img_%d.jpg', detection_result_common_name, imgIdx);
%         print('-djpeg','-r150',fullfile(SAVE_PATH, save_name));
        % save_name = sprintf('%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_imgIdx_%d.png',...
        % DATA_SET, CLASS, TYPE, detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs),imgIdx);
        % print('-dpng','-r150',['Result/' LOWER_CASE_CLASS '_' TYPE '/' save_name])
        
        dwot_save_detection(bbsNMS_clip_per_template_nms_mat_nms, SAVE_PATH, detection_result_file, ...
                           img_file_name, false, 1); % save mode != 0 to save template index
    end
    
    npos = npos + sum(~gt{imgIdx}.diff);
end


close all;  % space plot casues problem when using different subplot grid

%% Vary NMS threshold
nms_thresholds = 0.2 : 0.05 : 0.6;
ap = zeros(numel(nms_thresholds),1);
ap_save_names = cell(numel(nms_thresholds),1);
for i = 1:numel(nms_thresholds)
    nms_threshold = nms_thresholds(i);
    ap(i) = dwot_analyze_and_visualize_3D_object_results(fullfile(SAVE_PATH, detection_result_file),...
        detectors, SAVE_PATH, param, DATA_PATH, CLASS, param.color_range, nms_threshold, false);
                        

    ap_save_names{i} = sprintf(['AP_%s_nms_%0.2f.png'],...
                        detection_result_common_name, nms_threshold);

     print('-dpng','-r150',fullfile(SAVE_PATH, ap_save_names{i}));
end

% If it runs on server copy to host
if ~isempty(server_id) && isempty(strmatch(server_id.num,'capri7'))
    for i = 1:numel(nms_thresholds)
        system(['scp ', fullfile(SAVE_PATH, ap_save_names{i}),...
            ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/Result/',...
            LOWER_CASE_CLASS '_' TEST_TYPE]);
    end
    system(['scp ' fullfile(SAVE_PATH, detection_result_file),...
        ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/Result/']);
    
    if param.proposal_tuning_mode > 1
        system(['scp ', fullfile(SAVE_PATH, ap_tuning_save_name),...
            ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/Result/',...
            LOWER_CASE_CLASS '_' TEST_TYPE]);
        system(['scp ' fullfile(SAVE_PATH, detection_tuning_result_file),...
            ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/Result/']);
    end
end


%% AP view

% detScore_view = cell2mat(detScore_view);
% fp_view = cell2mat(fp_view);
% tp_view = cell2mat(tp_view);
% % atp = cell2mat(atp);
% % afp = cell2mat(afp);
% 
% [sc, si] =sort(detScore_view,'descend');
% fpSort_view = cumsum(fp_view(si));
% tpSort_view = cumsum(tp_view(si));
% 
% 
% recall_view = tpSort_view/npos;
% precision_view = tpSort_view./(fpSort_view + tpSort_view);
% 
% % arecall = atpSort/npos;
% % aprecision = atpSort./(afpSort + atpSort);
% ap_view = VOCap(recall_view', precision_view');
% % aa = VOCap(arecall', aprecision');
% fprintf('AP View = %.4f\n', ap_view);
% 
% 
% clf;
% plot(recall, precision, 'r', 'LineWidth',3);
% hold on;
% plot(recall_view, precision_view, 'g', 'LineWidth',3);
% xlabel('Recall');
% % ylabel('Precision/Accuracy');
% % tit = sprintf('Average Precision = %.1f / Average Accuracy = %1.1f', 100*ap,100*aa);
% 
% tit = sprintf('Average Precision = %.3f View Precision = %.3f', 100*ap, 100*ap_view);
% title(tit);
% axis([0 1 0 1]);
% set(gcf,'color','w');
% save_name = sprintf('AP_view_nms_%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_N_IM_%d.png',...
%         DATA_SET, LOWER_CASE_CLASS, TYPE, detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs),N_IMAGE);
% 
% print('-dpng','-r150',['Result/' LOWER_CASE_CLASS '_' TYPE '/' save_name])
% 
% 
% %% Confusion Matrix visualization
% clf;
% confusion_rate = confusion_statistics;
% for view_idx = 1:n_views
%   confusion_rate(:,view_idx) = confusion_rate(:,view_idx) / sum(confusion_rate(:,view_idx));
% end
% confusion_rate(isnan(confusion_rate)) = 0;
% 
% imagesc(confusion_rate);
% colorbar;
% xlabel('Ground Truth Viewpoints');
% ylabel('Prediction Viewpoints');
% set(gcf,'color','w');
% save_name = sprintf('confusion_matrix_view_nms_%0.2f_%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_N_IM_%d.png',...
%         nms_threshold, DATA_SET, LOWER_CASE_CLASS, TYPE, detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs),N_IMAGE);
% 
% print('-dpng','-r150',['Result/' LOWER_CASE_CLASS '_' TYPE '/' save_name]);
% clf;