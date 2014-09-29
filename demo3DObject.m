% Tests average model
% The average template is generated using multiple renderings
% 
% see dwot_avg_model.m
DATA_SET = '3DObject';
DATA_PATH = '/home/chrischoy/Dataset/3DObject/';
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

% Computing Mode  = 0, CPU
%                 = 1, GPU
%                 = 2, Combined
COMPUTING_MODE = 1;
CLASS = 'bicycle';
TYPE = 'val';
mkdir('Result',[CLASS '_' TYPE]);

if COMPUTING_MODE > 0
  gdevice = gpuDevice(1);
  reset(gdevice);
  cos(gpuArray(1));
  
  % Debug
  param.gpu = gdevice;
end
daz = 45;
del = 30;
dyaw = 15;
dfov = 20;

azs = 0:daz:315; % azs = [azs , azs - 10, azs + 10];
els = 0:del:60;
fovs = 20;
yaws = -45:dyaw:45;
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
detection_threshold = 80;
n_proposals = 10;
dwot_get_default_params;

% Get all possible sub-classes
model_paths = 'Mesh/Bicycle/';
% model_names = {'road_bike','road_bike_2','road_bike_3','fixed_gear_road_bike','bmx_bike','brooklyn_machine_works_bike', 'glx_bike'};
model_names = {'road_bike'}; % ,'bmx_bike','brooklyn_machine_works_bike'};
detector_model_name = ['each_' strjoin(strrep(model_names, '/','_'),'_')];
model_files = cellfun(@(x) [model_paths strrep([x '.3ds'], '/', '_')], model_names, 'UniformOutput', false);


param.models_path = model_paths;


if exist('renderer','var')
  renderer.delete();
  clear renderer;
end


if ~exist('renderer','var')
  renderer = Renderer();
  if ~renderer.initialize(model_files, 700, 700, 0, 0, 0, 0, 25)
    error('fail to load model');
  end
end

detector_name = sprintf('%s_init_%d_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d.mat',...
    CLASS, param.template_initialization_mode, detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs));

if exist(detector_name,'file')
  load(detector_name);
else
  [detectors] = dwot_make_detectors_grid(renderer, azs, els, yaws, fovs, 1:length(model_files), CLASS, param, visualize_detector);
  [detectors, detector_table]= dwot_make_table_from_detectors(detectors);
  if sum(cellfun(@(x) isempty(x), detectors))
    error('Detector Not Completed');
  end
  eval(sprintf(['save ' detector_name ' detectors detector_table']));
end

% For Debuggin purpose only
param.detectors              = detectors;
param.detect_pyramid_padding = 10;

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

% extract ground truth objects
npos = 0;
tp = cell(1,N_IMAGE);
fp = cell(1,N_IMAGE);
% atp = cell(1,N_IMAGE);
% afp = cell(1,N_IMAGE);
detScore = cell(1,N_IMAGE);
detectorId = cell(1,N_IMAGE);
detIdx = 0;


% extract ground truth objects
tp_prop = cell(1,N_IMAGE);
fp_prop = cell(1,N_IMAGE);
% atp = cell(1,N_IMAGE);
% afp = cell(1,N_IMAGE);
detScore_prop = cell(1,N_IMAGE);
detectorId_prop = cell(1,N_IMAGE);
detIdx_prop = 0;


% 138
% 256
for imgIdx = 1:(N_IMAGE)
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
    bbsNMS = esvm_nms(bbsAllLevel,0.5);
    
    nDet = size(bbsNMS,1);
    bbsNMS_clip = clip_to_image(bbsNMS, [1 1 imSz(2) imSz(1)]);
    [bbsNMS_clip, tp{imgIdx}, fp{imgIdx}, detScore{imgIdx}, ~] = dwot_compute_positives(bbsNMS_clip, gt{imgIdx}, param);
    
    if nDet > 0
      bbsNMS(:,9) = bbsNMS_clip(:,9);
    end
    
    if visualize_detection && ~isempty(clsinds)
      % figure(2);
      dwot_draw_overlap_detection(im, bbsNMS, renderings, n_proposals, 50, visualize_detection);
      drawnow;
      save_name = sprintf('%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_imgIdx_%d.png',...
        DATA_SET, CLASS, TYPE, detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs),imgIdx);
      print('-dpng','-r100',['Result/' CLASS '_' TYPE '/' save_name])
      
      %  waitforbuttonpress;
    end
    
    
%     n_mcmc = min(n_proposals, size(bbsNMS,1));
    
%     [hog_region_pyramid, im_region] = dwot_extract_hog(hog, scales, detectors, bbsNMS(1:n_mcmc,:), param, im);
%     [best_proposals, detectors, detector_table] = dwot_binary_search_proposal_region(hog_region_pyramid, im_region, detectors, detector_table, renderer, param, im);
%     [best_proposals] = dwot_mcmc_proposal_region(renderer, hog_region_pyramid, im_region, detectors, param, im);

%     for proposal_idx = 1:n_mcmc
%       subplot(121);
%       dwot_draw_overlap_detection(im, bbsNMS, renderings, n_mcmc, 50, true);
%       subplot(122);
%       bbox = best_proposals{proposal_idx}.image_bbox;
%       bbox(12) = best_proposals{proposal_idx}.score;
%       dwot_draw_overlap_detection(im, bbox, best_proposals{proposal_idx}.rendering_image, n_mcmc, 50, true);
%       axis equal;
%       fprintf('press any button to continue\n');
%       waitforbuttonpress
%     end
    


    % dwot_bfgs_proposal_region(hog_region_pyramid, im_region, detectors, param); 
%     mcmc_score = cellfun(@(x) x.score, best_proposals);
%     mcmc_score ./ bbsNMS_clip(1:n_mcmc,end)'
%     bbsNMS_clip(1:n_mcmc,end) = mcmc_score';
    
%     bbsNMS_proposal = bbsNMS;
%     for region_idx = 1:n_mcmc
%       bbsNMS_proposal(region_idx,1:4) = best_proposals{region_idx}.image_bbox;
%       bbsNMS_proposal(region_idx,12) = best_proposals{region_idx}.score;
%     end
%     [~, o] = sort(bbsNMS_proposal(:,12),'descend');
%     bbsNMS_proposal = bbsNMS_proposal(o,:); 
%     bbsNMS_proposal_clip = clip_to_image(bbsNMS_proposal, [1 1 imSz(2) imSz(1)]);
%     
%     fprintf(' time to mcmc: %0.4f', toc(imgTic));
% %    
% 
%     [bbsNMS_proposal_clip, tp_prop{imgIdx}, fp_prop{imgIdx}, ~] = dwot_compute_positives(bbsNMS_proposal_clip, gt(imgIdx), param);
%     fprintf(' time : %0.4f\n', toc(imgTic));

    if nDet > 0
      detectorId{imgIdx} = bbsNMS(:,11)';
%       detScore{imgIdx} = bbsNMS(:,end)';
%       detectorId_prop{imgIdx} = bbsNMS_proposal_clip(:,11)';
%       detScore_prop{imgIdx} = bbsNMS_proposal_clip(:,end)';
    else
      detectorId{imgIdx} = [];
%       detScore{imgIdx} = [];
%       detectorId_prop{imgIdx} = [];
%       detScore_prop{imgIdx} = [];
    end
    % if visualize
%     if visualize_detection && ~isempty(clsinds)
%       figure(3);
%       
%       bbsNMS_proposal(:,9) = bbsNMS_proposal_clip(:,9);
%       dwot_draw_overlap_detection(im, bbsNMS_proposal, renderings, n_mcmc, 50, visualize_detection);
% 
%       % disp('Press any button to continue');
%       
%       save_name = sprintf('%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_imgIdx_%d_binary.png',...
%         CLASS,TYPE, average_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs),imgIdx);
%       print('-dpng','-r100',['Result/' CLASS '_' TYPE '/' save_name])
%       
%       % waitforbuttonpress;
%     end
      
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
detectorId = cell2mat(detectorId);

[sc, si] =sort(detScore,'descend');
fpSort = cumsum(fp(si));
tpSort = cumsum(tp(si));

% atpSort = cumsum(atp(si));
% afpSort = cumsum(afp(si));

detectorIdSort = detectorId(si);

recall = tpSort/npos;
precision = tpSort./(fpSort + tpSort);

% arecall = atpSort/npos;
% aprecision = atpSort./(afpSort + atpSort);
ap = VOCap(recall', precision');
% aa = VOCap(arecall', aprecision');
fprintf('AP = %.4f\n', ap);

clf;
plot(recall, precision, 'r', 'LineWidth',3);
% hold on;
% plot(arecall, aprecision, 'g', 'LineWidth',3);
xlabel('Recall');
% ylabel('Precision/Accuracy');
% tit = sprintf('Average Precision = %.1f / Average Accuracy = %1.1f', 100*ap,100*aa);

tit = sprintf('Average Precision = %.1f', 100*ap);
title(tit);
axis([0 1 0 1]);
set(gcf,'color','w');
save_name = sprintf('AP_%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_N_IM_%d.png',...
        DATA_SET, CLASS, TYPE, detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs),N_IMAGE);

print('-dpng','-r150',['Result/' CLASS '_' TYPE '/' save_name])





% detScore_prop = cell2mat(detScore_prop);
% fp_prop = cell2mat(fp_prop);
% tp_prop = cell2mat(tp_prop);
% % atp = cell2mat(atp);
% % afp = cell2mat(afp);
% detectorId_prop = cell2mat(detectorId_prop);
% 
% [sc, si] =sort(detScore_prop,'descend');
% fpSort_prop = cumsum(fp_prop(si));
% tpSort_prop = cumsum(tp_prop(si));
% 
% % atpSort = cumsum(atp(si));
% % afpSort = cumsum(afp(si));
% 
% detectorIdSort_prop = detectorId_prop(si);
% 
% recall_prop = tpSort_prop/npos;
% precision_prop = tpSort_prop./(fpSort_prop + tpSort_prop);
% 
% % arecall = atpSort/npos;
% % aprecision = atpSort./(afpSort + atpSort);
% ap_prop = VOCap(recall_prop', precision_prop');
% % aa = VOCap(arecall', aprecision');
% fprintf('AP = %.4f\n', ap_prop);
% 
% clf;
% plot(recall_prop, precision_prop, 'r', 'LineWidth',3);
% % hold on;
% % plot(arecall, aprecision, 'g', 'LineWidth',3);
% xlabel('Recall');
% % ylabel('Precision/Accuracy');
% % tit = sprintf('Average Precision = %.1f / Average Accuracy = %1.1f', 100*ap,100*aa);
% 
% tit = sprintf('Average Precision = %.1f', 100*ap_prop);
% title(tit);
% axis([0 1 0 1]);
% set(gcf,'color','w');
% save_name = sprintf('AP_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_N_IM_%d_binary.png',...
%         CLASS, TYPE, model_names{1}, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs),N_IMAGE);
% 
% print('-dpng','-r150',['Result/' CLASS '_' TYPE '/' save_name]);


%% Cleanup Memory
if exist('renderer','var')
  renderer.delete();
  clear renderer;
end
