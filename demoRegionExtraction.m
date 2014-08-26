VOC_PATH = '/home/chrischoy/Dataset/VOCdevkit/';
if ismac
  VOC_PATH = '~/dataset/VOCdevkit/';
end

addpath('HoG');
addpath('HoG/features');
addpath('Util');
addpath('DecorrelateFeature/');
addpath('../MatlabRenderer/');
addpath('../MatlabCUDAConv/');
addpath(VOC_PATH);
addpath([VOC_PATH, 'VOCcode']);

% Computing Mode  = 0, CPU
%                 = 1, GPU
%                 = 2, Combined
COMPUTING_MODE = 0;
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
del = 20;
dfov = 10;
dyaw = 10;

azs = 0:45:315; % azs = [azs , azs - 10, azs + 10];
els = 0:20:20;
fovs = [25];
yaws = [-10:10:10];
n_cell_limit = [190];
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
detection_threshold = 120;
n_proposals = 1;

models_path = {'Mesh/Bicycle/road_bike'};
models_name = cellfun(@(x) strrep(x, '/', '_'), models_path, 'UniformOutput', false);

dwot_get_default_params;
param.models_path = models_path;

detector_name = sprintf('%s_%d_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d.mat',...
    CLASS, numel(models_path), n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs));

if exist(detector_name,'file')
  load(detector_name);
else
  detectors = dwot_make_detectors([models_path{1} '.3ds'], azs, els, yaws, fovs, param, visualize_detector);
  if sum(cellfun(@(x) isempty(x), detectors))
    error('Detector Not Completed');
  end
  eval(sprintf(['save ' detector_name ' detectors']));
end


% For Debuggin purpose only
param.detectors              = detectors;
param.detect_pyramid_padding = 10;

renderings = cellfun(@(x) x.rendering, detectors, 'UniformOutput', false);

if COMPUTING_MODE == 0
  templates = cellfun(@(x) single(x.whow), detectors,'UniformOutput',false);
elseif COMPUTING_MODE == 1
  templates = cellfun(@(x) (single(x.whow(end:-1:1,end:-1:1,:))), detectors,'UniformOutput',false);
elseif COMPUTING_MODE == 2
  templates = cellfun(@(x) gpuArray(single(x.whow(end:-1:1,end:-1:1,:))), detectors,'UniformOutput',false);
  templates_cpu = cellfun(@(x) single(x.whow), detectors,'UniformOutput',false);
else
  error('Computing mode undefined');
end

curDir = pwd;
eval(['cd ' VOC_PATH]);
VOCinit;
eval(['cd ' curDir]);

% load dataset
[gtids,t] = textread(sprintf(VOCopts.imgsetpath,[CLASS '_' TYPE]),'%s %d');

N_IMAGE = length(gtids);

% extract ground truth objects
npos = 0;
tp = cell(1,N_IMAGE);
fp = cell(1,N_IMAGE);
% atp = cell(1,N_IMAGE);
% afp = cell(1,N_IMAGE);
detScore = cell(1,N_IMAGE);
detectorId = cell(1,N_IMAGE);
detIdx = 0;

gt(length(gtids))=struct('BB',[],'diff',[],'det',[]);
% 138
% 256
for imgIdx = 312:N_IMAGE
    fprintf('%d/%d ',imgIdx,N_IMAGE);
    imgTic = tic;
    % read annotation
    recs(imgIdx)=PASreadrecord(sprintf(VOCopts.annopath,gtids{imgIdx}));
    
    clsinds = strmatch(CLASS,{recs(imgIdx).objects(:).class},'exact');
    gt(imgIdx).BB=cat(1,recs(imgIdx).objects(clsinds).bbox)';
    gt(imgIdx).diff=[recs(imgIdx).objects(clsinds).difficult];
    gt(imgIdx).det=false(length(clsinds),1);
    
    if isempty(clsinds)
      continue;
    end
    
    im = imread([VOCopts.datadir, recs(imgIdx).imgname]);
    imSz = size(im);
    if COMPUTING_MODE == 0
      [bbsNMS, hog, scales] = dwot_detect( im, templates, param);
      % [hog_region_pyramid, im_region] = dwot_extract_region_conv(im, hog, scales, bbsNMS, param);
      % [bbsNMS_MCMC] = dwot_mcmc_proposal_region(im, hog, scale, hog_region_pyramid, param);
    elseif COMPUTING_MODE == 1
      % [bbsNMS ] = dwot_detect_gpu_and_cpu( im, templates, templates_cpu, param);
      [bbsNMS, hog, scales] = dwot_detect_gpu( im, templates, param);
    elseif COMPUTING_MODE == 2
      [bbsNMS, hog, scales] = dwot_detect_combined( im, templates, templates_cpu, param);
    else
      error('Computing Mode Undefined');
    end
    fprintf(' time to convolution: %0.4f', toc(imgTic));
    
    bbsNMS_clip = clip_to_image(bbsNMS, [1 1 imSz(2) imSz(1)]);

    if visualize_detection && ~isempty(clsinds)
      figure(2);
      dwot_draw_overlap_detection(im, bbsNMS_clip, renderings, n_proposals, 50, visualize_detection);
      drawnow;
      %  waitforbuttonpress;
    end
    n_mcmc = min(n_proposals, size(bbsNMS,1));
    [hog_region_pyramid, im_region] = dwot_extract_hog(hog, scales, templates, bbsNMS(1:n_mcmc,:), param, im);
    [best_proposals] = dwot_binary_search_proposal_region(hog_region_pyramid, im_region, detectors, param, im);
    % [best_proposals] = dwot_mcmc_proposal_region(hog_region_pyramid, im_region, detectors, param, im);

    for proposal_idx = 1:n_mcmc
      subplot(121);
      dwot_draw_overlap_detection(im, bbsNMS_clip, renderings, n_mcmc, 50, true);
      subplot(122);
      dwot_draw_overlap_detection(im, best_proposals{proposal_idx}.image_bbox, best_proposals{proposal_idx}.rendering_image, n_mcmc, 50, true);
      fprintf('press any button to continue\n');
      % waitforbuttonpress
    end
    

    bbsNMS_clip = clip_to_image(bbsNMS_clip, [1 1 imSz(2) imSz(1)]);
    % dwot_bfgs_proposal_region(hog_region_pyramid, im_region, detectors, param); 
    mcmc_score = cellfun(@(x) x.score, best_proposals);
    mcmc_score ./ bbsNMS_clip(1:n_mcmc,end)'
    bbsNMS_clip(1:n_mcmc,end) = mcmc_score';
    
    for region_idx = 1:n_mcmc
      bbsNMS_clip(region_idx,1:4) = best_proposals{region_idx}.image_bbox;
    end
    
    fprintf(' time to mcmc: %0.4f', toc(imgTic));
    nDet = size(bbsNMS_clip,1);

    if nDet > 0
      detectorId{imgIdx} = bbsNMS_clip(:,11)';
      detScore{imgIdx} = bbsNMS_clip(:,end)';
    else
      detectorId{imgIdx} = [];
      detScore{imgIdx} = [];
    end
    
    [bbsNMS_clip, tp{imgIdx}, fp{imgIdx}, gt(imgIdx)] = dwot_compute_positives(bbsNMS_clip, gt(imgIdx), param);
    
    fprintf(' time : %0.4f\n', toc(imgTic));

    % if visualize
    if visualize_detection && ~isempty(clsinds)
      dwot_draw_overlap_detection(im, bbsNMS_clip, renderings, 5, 50, visualize_detection);

      % disp('Press any button to continue');
      
      % save_name = sprintf('%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_imgIdx_%d.jpg',...
      %   CLASS,TYPE,model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs),imgIdx);
      % print('-djpeg','-r100',['Result/' CLASS '_' TYPE '/' save_name])
      
      % waitforbuttonpress;
    end
      
    npos = npos + sum(~gt(imgIdx).diff);
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
save_name = sprintf('AP_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d.png',...
        CLASS, TYPE, models_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs));

print('-dpng','-r150',['Result/' CLASS '_' TYPE '/' save_name])
