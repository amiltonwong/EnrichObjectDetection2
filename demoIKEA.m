% Tests average model
% The average template is generated using multiple renderings
% 
% see dwot_avg_model.m
DATA_SET = 'IKEA';

[~, sys_result] = system('hostname');
server_id = regexp(sys_result, '^napoli(?<num>\d+).*','names');
if isempty(server_id)
  DATA_PATH = '/home/chrischoy/Dataset/';
else
  DATA_PATH = '/scratch/chrischoy/Dataset/';
end

if ismac
  DATA_PATH = '~/dataset/';
end
DATA_PATH = fullfile(DATA_PATH, DATA_SET);

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

% Computing Mode  = 0, CPU
%                 = 1, GPU
%                 = 2, Combined
COMPUTING_MODE = 1;
CLASS = 'Chair';
SUB_CLASS = [];
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

daz = 15;
del = 15;
dyaw = 15;
dfov = 20;

azs = 0:daz:345; 
% azs = [azs , azs - 10, azs + 10];
els = 0:del:45;
fovs = [25];
yaws = 0;
% yaws = -15:dyaw:15;
n_cell_limit = [150];
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
model_pahts = [DATA_PATH '/data/models/IKEA/'];
[ model_names, file_paths ]= dwot_get_cad_models_substr(model_pahts, CLASS, SUB_CLASS, {'3ds','obj'});

% model_names = {'road_bike','bmx_bike', 'glx_bike'};
% model_names = {'road_bike','road_bike_2','road_bike_3','fixed_gear_road_bike','bmx_bike','brooklyn_machine_works_bike', 'glx_bike'};
% model_names = {'road_bike'}; % ,'bmx_bike','brooklyn_machine_works_bike'};
% AccordIdx = ismember(model_names,'Honda-Accord-3');
% model_names = model_names(AccordIdx);
% file_paths = file_paths(AccordIdx);

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
annotation = load([DATA_PATH '/pos_bbox.mat']);
% For IKEA dataset, there are images without rotation matrix. These are not
% counted as dataset.
non_empty_bbox = cellfun(@(x) ~isempty(x),(annotation.bbox));
bbox = annotation.bbox(non_empty_bbox);
gt = annotation.pos(non_empty_bbox);

N_IMAGE = length(gt);

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
% atp = cell(1,N_IMAGE);
% afp = cell(1,N_IMAGE);
detScore_prop = cell(1,N_IMAGE);
detectorId_prop = cell(1,N_IMAGE);

tp_per_template = cell(1,numel(templates));
fp_per_template = cell(1,numel(templates));

tp_per_template_view = cell(1,numel(templates));
fp_per_template_view = cell(1,numel(templates));

% 138
% 256
for imgIdx = floor(N_IMAGE/4):ceil(N_IMAGE/2)
    fprintf('%d/%d ',imgIdx,N_IMAGE);
    imgTic = tic;
    
    
    im = imread(fullfile(DATA_PATH,gt(imgIdx).im));
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
    

    bbsNMS = esvm_nms(bbsAllLevel,0.5);
    
    nDet = size(bbsNMS,1);
    bbsNMS_clip = clip_to_image(bbsNMS, [1 1 imSz(2) imSz(1)]);
    
    gt_mod = gt(imgIdx);
    BB = [];
    for obj_idx = 1:numel(gt_mod.gt_info)
      if ~isempty(regexpi(gt_mod.gt_info(obj_idx).type,CLASS))
        BB = [BB; gt_mod.gt_info(obj_idx).bbox];
      end
    end  
    gt_mod.diff = zeros(1,size(BB,1));
    gt_mod.BB = BB';
    gt_mod.det = zeros(1,size(BB,1));

    [bbsNMS_clip, tp{imgIdx}, fp{imgIdx}, detScore{imgIdx}, ~] = dwot_compute_positives(bbsNMS_clip, gt_mod, param);
    
    if nDet > 0
%       detScore{imgIdx} = bbsNMS_clip(:,end)';
      bbsNMS(:,9) = bbsNMS_clip(:,9);
    end
    
    if visualize_detection && ~isempty(clsinds)
      % figure(2);
      dwot_draw_overlap_detection(im, bbsNMS, renderings, n_proposals, 50, visualize_detection);
      drawnow;
      save_name = sprintf('%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_imgIdx_%d.jpg',...
        DATA_SET, LOWER_CASE_CLASS, TYPE, detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs),imgIdx);
      print('-djpeg','-r150',['Result/' LOWER_CASE_CLASS '_' TYPE '/' save_name]);
      
      %  waitforbuttonpress;
    end
    
    npos = npos + sum(~gt_mod.diff);
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

[sc, si] =sort(detScore,'descend');
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

% detScore_view = cell2mat(detScore_view);
% fp_view = cell2mat(fp_view);
% tp_view = cell2mat(tp_view);
% atp = cell2mat(atp);
% afp = cell2mat(afp);

% [sc, si] =sort(detScore_view,'descend');
% fpSort_view = cumsum(fp_view(si));
% tpSort_view = cumsum(tp_view(si));


% recall_view = tpSort_view/npos;
% precision_view = tpSort_view./(fpSort_view + tpSort_view);

% arecall = atpSort/npos;
% aprecision = atpSort./(afpSort + atpSort);
% ap_view = VOCap(recall_view', precision_view');
% aa = VOCap(arecall', aprecision');
% fprintf('AP View = %.4f\n', ap_view);


clf;
plot(recall, precision, 'r', 'LineWidth',3);
hold on;
% plot(recall_view, precision_view, 'g', 'LineWidth',3);
xlabel('Recall');
% ylabel('Precision/Accuracy');
% tit = sprintf('Average Precision = %.1f / Average Accuracy = %1.1f', 100*ap,100*aa);

tit = sprintf('Average Precision = %.1f', 100*ap);
% tit = sprintf('Average Precision = %.1f View Precision = %.1f', 100*ap, 100*ap_view);
title(tit);
axis([0 1 0 1]);
set(gcf,'color','w');
save_name = sprintf('AP_view_nms_to_views_%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_N_IM_%d.png',...
        DATA_SET, LOWER_CASE_CLASS, TYPE, detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs),N_IMAGE);

print('-dpng','-r150',['Result/' LOWER_CASE_CLASS '_' TYPE '/' save_name])


% figure(3);
% for template_idx = 1:numel(templates)
%   subplot(131);
%   cla;
%   [y_tp,x_tp] = hist(tp_per_template{template_idx},5);
%   tp_area = trapz(x_tp,y_tp);
%   bar(x_tp,y_tp);
%   hold on;
%   [y_fp,x_fp] = hist(fp_per_template{template_idx});
%   fp_area = trapz(x_fp,y_fp);
%   bar(x_fp,y_fp);
%   
%   h = findobj(gca,'Type','patch');
%   set(h(1),'Facecolor',[1 0 0],'EdgeColor','k','FaceAlpha',0.5);
%   set(h(2),'Facecolor',[0 0 1],'EdgeColor','k');
%   
%   subplot(132);
%   cla;
%   bar(x_tp,y_tp/tp_area,'k');
%   hold on;
%   bar(x_fp,y_fp/fp_area,'r');
%   
%   h = findobj(gca,'Type','patch');
%   set(h(1),'Facecolor',[1 0 0],'EdgeColor','k','FaceAlpha',0.5);
%   set(h(2),'Facecolor',[0 0 1],'EdgeColor','k');
%   
%   
%   subplot(133);
%   imagesc(renderings{template_idx});  
%   axis equal; axis off;
%   drawnow;
%   
%   set(gcf,'color','w');
%   save_name = sprintf('hist_%s_%s_%s_%s_lim_%d_lam_%0.4f_ID_%d.png',...
%           DATA_SET, LOWER_CASE_CLASS, TYPE, detector_model_name, n_cell_limit, lambda, template_idx);
% 
%   print('-dpng','-r100',['Result/' CLASS '_' TYPE '/' save_name])
% end


%% Cleanup Memory
if exist('renderer','var')
  renderer.delete();
  clear renderer;
end
