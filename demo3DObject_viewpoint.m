% Tests average model
% The average template is generated using multiple renderings
% 
% see dwot_avg_model.m
DATA_SET = '3DObject';

[~, sys_result] = system('hostname');
server_id = regexp(sys_result, '^napoli(?<num>\d+).*','names');
if isempty(server_id)
  DATA_PATH = '/home/chrischoy/Dataset/3DObject/';
else
  DATA_PATH = '/scratch/chrischoy/Dataset/3DObject/';
end

if ismac
  DATA_PATH = '~/dataset/3DObject';
end

% VOC_PATH = '/home/chrischoy/Dataset/VOCdevkit/';
% if ismac
%   VOC_PATH = '~/dataset/VOCdevkit/';
% end

addpath('HoG');
addpath('HoG/features');
addpath('Util');
addpath('DecorrelateFeature/');
addpath('../MatlabRenderer/');
addpath('../MatlabCUDAConv/');
addpath(DATA_PATH);
addpath('3rdParty/SpacePlot');

% Computing Mode  = 0, CPU
%                 = 1, GPU
%                 = 2, Combined
COMPUTING_MODE = 1;
CLASS = 'Car';
SUB_CLASS = 'Sedan';
LOWER_CASE_CLASS = lower(CLASS);
TYPE = 'val';
mkdir('Result',[LOWER_CASE_CLASS '_' TYPE]);

if COMPUTING_MODE > 0
  gdevice = gpuDevice(1);
  reset(gdevice);
  cos(gpuArray(1));
  
  % Debug
  param.gpu = gdevice;
end

daz = 45;
del = 15;
dyaw = 15;
dfov = 20;

azs = 0:daz:315; 
azs = [azs , azs - 10, azs + 10];
els = 0:del:45;
fovs = [20 40 60];
yaws = 0;
% yaws = -15:dyaw:15;
n_cell_limit = [200];
lambda = [0.015];

% azs = 0:15:345
% els = 0 : 15 :30
% fovs = 25
% yaws = -45:15:45
% n_cell_limit = 135
% lambda = 0.015

visualize_detection = true;
visualize_detector = false;
% visualize = false;

sbin = 4;
n_level = 10;
detection_threshold = 70;
n_proposals = 5;
dwot_get_default_params;

% Get all possible sub-classes
% model_paths = fullfile('Mesh', CLASS);
% model_names = {'road_bike'}; % ,'bmx_bike','brooklyn_machine_works_bike'};
% detector_model_name = ['init_' num2str(param.template_initialization_mode) '_each_' strjoin(strrep(model_names, '/','_'),'_')];
% model_files = cellfun(@(x) [model_paths strrep([x '.3ds'], '/', '_')], model_names, 'UniformOutput', false);



% Get all possible sub-classes
model_paths = fullfile('Mesh', CLASS);
[ model_names, file_paths ] = dwot_get_cad_models('Mesh', CLASS, SUB_CLASS, {'3ds'});

% model_names = {'road_bike','bmx_bike', 'glx_bike'};
% model_names = {'road_bike','road_bike_2','road_bike_3','fixed_gear_road_bike','bmx_bike','brooklyn_machine_works_bike', 'glx_bike'};
% model_names = {'road_bike'}; % ,'bmx_bike','brooklyn_machine_works_bike'};
AccordIdx = ismember(model_names,'Honda-Accord-3');
model_names = model_names(AccordIdx);
file_paths = file_paths(AccordIdx);

% detector name
[ detector_model_name ] = dwot_get_detector_name(CLASS, SUB_CLASS, model_names, param);
detector_name = sprintf('%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d.mat',...
     LOWER_CASE_CLASS,  detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs));

param.model_paths = file_paths;

if exist('renderer','var')
  renderer.delete();
  clear renderer;
end

if ~exist('renderer','var')
  renderer = Renderer();
  if ~renderer.initialize(file_paths, 700, 700, 0, 0, 0, 0, 25)
    error('fail to load model');
  end
end

if exist(detector_name,'file')
  load(detector_name);
else
  [detectors] = dwot_make_detectors_grid(renderer, azs, els, yaws, fovs, 1:length(model_names), LOWER_CASE_CLASS, param, visualize_detector);
  [detectors, detector_table]= dwot_make_table_from_detectors(detectors);
  if sum(cellfun(@(x) isempty(x), detectors))
    error('Detector Not Completed');
  end
  eval(sprintf(['save ' detector_name ' detectors detector_table']));
end

% For Debuggin purpose only
param.detectors              = detectors;
param.detect_pyramid_padding = 10;
param.min_overlap = 0.5;


renderings = cellfun(@(x) x.rendering_image, detectors, 'UniformOutput', false);

if COMPUTING_MODE == 0
  templates = cellfun(@(x) (single(x.whow)), detectors,'UniformOutput',false);
elseif COMPUTING_MODE == 1
  templates = cellfun(@(x) (single(x.whow(end:-1:1,end:-1:1,:))), detectors,'UniformOutput',false);
elseif COMPUTING_MODE == 2
  templates = cellfun(@(x) (single(x.whow(end:-1:1,end:-1:1,:))), detectors,'UniformOutput',false);
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
N_IMAGE = ceil(N_IMAGE/4);

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

tp_per_template = cell(1,numel(templates));
fp_per_template = cell(1,numel(templates));

n_views = 8;
confusion_statistics = zeros(n_views, n_views);

for imgIdx = 1:N_IMAGE
    fprintf('%d/%d ',imgIdx,N_IMAGE);
    imgTic = tic;

    im = imread([image_path{imgIdx}]);
    imSz = size(im);
    if COMPUTING_MODE == 0
      [bbsAllLevel, hog, scales] = dwot_detect( im, templates, param);
      % [hog_region_pyramid, im_region] = dwot_extract_region_conv(im, hog, scales, bbsNMS, param);
      % [bbsNMS_MCMC] = dwot_mcmc_proposal_region(im, hog, scale, hog_region_pyramid, param);
    elseif COMPUTING_MODE == 1
      % [bbsNMS ] = dwot_detect_gpu_and_cpu( im, templates, templates_cpu, param);
      [bbsAllLevel, hog, scales] = dwot_detect_gpu( im, templates, param);
    elseif COMPUTING_MODE == 2
      [bbsAllLevel, hog, scales] = dwot_detect_combined( im, templates, templates_cpu, param);
    else
      error('Computing Mode Undefined');
    end
    fprintf(' convolution time : %0.4f\n', toc(imgTic));
    
    % Automatically sort them according to the score and apply NMS
    % bbsNMS = esvm_nms(bbsAllLevel,0.5);
    % Per Viewpoint NMS
    bbsNMS_per_template_nms = dwot_per_view_nms(bbsAllLevel,0.5);
    bbsNMS_per_template_nms_mat = cell2mat(bbsNMS_per_template_nms);
    bbsNMS_clip_per_template_mat = clip_to_image(bbsNMS_per_template_nms_mat, [1 1 imSz(2) imSz(1)]);
    
    % Compute overlap of per view nms detections
    [bbsNMS_clip_per_template_mat, ~, ~, ~] = dwot_compute_positives_view(bbsNMS_clip_per_template_mat, gt{imgIdx}, detectors, param);
    bbsNMS_per_template_nms_mat(:,9) = bbsNMS_clip_per_template_mat(:,9); % copy overlap to original non-clipped detection
    
    % Gather statistics
    [tp_per_template, fp_per_template] = dwot_gather_viewpoint_statistics(tp_per_template, fp_per_template, bbsNMS_clip_per_template_mat, 0.75);
    
    % NMS again
    [bbsNMS_clip_per_template_nms_mat_nms, nms_idx] = esvm_nms(bbsNMS_clip_per_template_mat,0.5);
    bbsNMS_per_template_nms_mat_nms = bbsNMS_per_template_nms_mat(nms_idx,:);
    
    % [bbsNMS_clip_per_template, tp_view{imgIdx}, fp_view{imgIdx}, ~] = dwot_compute_positives_view(bbsNMS_clip_per_template_mat, gt{imgIdx}, detectors, param);
    % Applying NMS dramatically reduce false positives
    [bbsNMS_clip_per_template_nms_mat_nms, tp{imgIdx}, fp{imgIdx}, detScore{imgIdx}, ~] = dwot_compute_positives(bbsNMS_clip_per_template_nms_mat_nms, gt{imgIdx}, param);

    % dwot_compute_positives_view will return GT index at the 10th column of bbsNMS
    [bbsNMS_clip_per_template_nms_mat_nms, tp_view{imgIdx}, fp_view{imgIdx}, detScore_view{imgIdx}, ~] = dwot_compute_positives_view(bbsNMS_clip_per_template_nms_mat_nms, gt{imgIdx}, detectors, param);
    
    % Confusion Matrix
    confusion_statistics = dwot_gather_confusion_statistics(confusion_statistics, detectors, gt{imgIdx}, bbsNMS_clip_per_template_nms_mat_nms, n_views);

    if visualize_detection
      tpIdx = bbsNMS_clip_per_template_nms_mat_nms(:, 9) > param.min_overlap;
      % Original images
      subplot(221);
      imagesc(im); axis off; axis equal;
      
      % True positives
      subplot(222);
      dwot_draw_overlap_detection(im, bbsNMS_per_template_nms_mat_nms(tpIdx,:), renderings, inf, 50, visualize_detection, [0.2, 0.8, 0] );
      
      % False positives
      subplot(223);
      dwot_draw_overlap_detection(im, bbsNMS_per_template_nms_mat_nms(~tpIdx,:), renderings, n_proposals, 50, visualize_detection);
      
      drawnow;
      spaceplots();
      save_name = sprintf('%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_imgIdx_%d.png',...
        DATA_SET, CLASS, TYPE, detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs),imgIdx);
      print('-dpng','-r150',['Result/' LOWER_CASE_CLASS '_' TYPE '/' save_name])
    end
    
    npos = npos + sum(~gt{imgIdx}.diff);
end

if isfield(param,'renderer')
  param.renderer.delete();
  param.renderer = [];
end

detScore = cell2mat(detScore);
fp = cell2mat(fp);
tp = cell2mat(tp);
% atp = cell2mat(atp);
% afp = cell2mat(afp);
% detectorId = cell2mat(detectorId);

[~, si] =sort(detScore,'descend');
fpSort = cumsum(fp(si));
tpSort = cumsum(tp(si));

% atpSort = cumsum(atp(si));
% afpSort = cumsum(afp(si));

% detectorIdSort = detectorId(si);

recall = tpSort/npos;
precision = tpSort./(fpSort + tpSort);

% arecall = atpSort/npos;
% aprecision = atpSort./(afpSort + atpSort);
ap = VOCap(recall', precision');
% aa = VOCap(arecall', aprecision');
fprintf('AP = %.4f\n', ap);



%% AP view

detScore_view = cell2mat(detScore_view);
fp_view = cell2mat(fp_view);
tp_view = cell2mat(tp_view);
% atp = cell2mat(atp);
% afp = cell2mat(afp);

[sc, si] =sort(detScore_view,'descend');
fpSort_view = cumsum(fp_view(si));
tpSort_view = cumsum(tp_view(si));


recall_view = tpSort_view/npos;
precision_view = tpSort_view./(fpSort_view + tpSort_view);

% arecall = atpSort/npos;
% aprecision = atpSort./(afpSort + atpSort);
ap_view = VOCap(recall_view', precision_view');
% aa = VOCap(arecall', aprecision');
fprintf('AP View = %.4f\n', ap_view);


clf;
plot(recall, precision, 'r', 'LineWidth',3);
hold on;
plot(recall_view, precision_view, 'g', 'LineWidth',3);
xlabel('Recall');
% ylabel('Precision/Accuracy');
% tit = sprintf('Average Precision = %.1f / Average Accuracy = %1.1f', 100*ap,100*aa);

tit = sprintf('Average Precision = %.1f View Precision = %.1f', 100*ap, 100*ap_view);
title(tit);
axis([0 1 0 1]);
set(gcf,'color','w');
save_name = sprintf('AP_view_nms_%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_N_IM_%d.png',...
        DATA_SET, LOWER_CASE_CLASS, TYPE, detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs),N_IMAGE);

print('-dpng','-r150',['Result/' LOWER_CASE_CLASS '_' TYPE '/' save_name])


confusion_rate = confusion_statistics;
for view_idx = 1:n_views
  confusion_rate(:,view_idx) = confusion_rate(:,view_idx) / sum(confusion_rate(:,view_idx));
end
confusion_rate(isnan(confusion_rate)) = 0;

imagesc(confusion_rate);
colorbar;
xlabel('Ground Truth Viewpoints');
ylabel('Prediction Viewpoints');
set(gcf,'color','w');
save_name = sprintf('confusion_matrix_view_nms_%0.2f_%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_N_IM_%d.png',...
        nms_threshold, DATA_SET, LOWER_CASE_CLASS, TYPE, detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs),N_IMAGE);

print('-dpng','-r150',['Result/' LOWER_CASE_CLASS '_' TYPE '/' save_name]);
clf;


%% Confusion Matrix visualization

%% Cleanup Memory
if exist('renderer','var')
  renderer.delete();
  clear renderer;
end
