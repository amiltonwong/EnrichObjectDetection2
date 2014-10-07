[~, sys_result] = system('hostname');
server_id = regexp(sys_result, '^napoli(?<num>\d+).*','names');
if isempty(server_id)
  VOC_PATH = '/home/chrischoy/Dataset/VOCdevkit/';
else
  VOC_PATH = '/scratch/chrischoy/Dataset/VOCdevkit/';
end

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
addpath('3rdParty/SpacePlot');
addpath('3rdParty/MinMaxSelection');

% Addpath doesn't work pwd in the code uses current directory so move to
% the directory.
curDir = pwd;
eval(['cd ' VOC_PATH]);
VOCinit;
eval(['cd ' curDir]);


DATA_SET = 'PASCAL';
COMPUTING_MODE = 1;
CLASS = 'Car';
% CLASS = 'Bicycle';
SUB_CLASS = [];
LOWER_CASE_CLASS = lower(CLASS);
TYPE = 'val';
mkdir('Result',[LOWER_CASE_CLASS '_' TYPE]);

if COMPUTING_MODE > 0
  gdevice = gpuDevice(1);
  reset(gdevice);
  cos(gpuArray(1));
end
daz = 45;
del = 20;
dfov = 10;
dyaw = 10;

azs = 0:15:345; % azs = [azs , azs - 10, azs + 10];
els = 0:20:20;
fovs = [25 50];
yaws = 0;
n_cell_limit = [300];
lambda = [0.015];
detection_threshold = 80;

% azs = 0:45:345
% els = 0 : 10 : 40
% fovs = [15, 45]
% yaws = -40:10:40
% n_cell_limit = 190
% lambda = 0.015

visualize_detection = true;
visualize_detector = false;
% visualize = false;

sbin = 4;
n_level = 10;
n_proposals = 5;

% Load models
% models_path = {'Mesh/Bicycle/road_bike'};
% models_name = cellfun(@(x) strrep(x, '/', '_'), models_path, 'UniformOutput', false);
[ model_names, model_paths ] = dwot_get_cad_models('Mesh', CLASS, [], {'3ds','obj'});

% models_to_use = {'bmx_bike',...
%               'fixed_gear_road_bike',...
%               'glx_bike',...
%               'road_bike'};
            
models_to_use = {'2012-VW-beetle-turbo',...
              'Kia_Spectra5_2006',...
              '2008-Jeep-Cherokee',...
              'Ford Ranger Updated',...
              'BMW_X1_2013',...
              'Honda_Accord_Coupe_2009',...
              'Porsche_911'};

use_idx = ismember(model_names,models_to_use);

model_names = model_names(use_idx);
model_paths = model_paths(use_idx);


skip_criteria = {'empty'};
skip_name = cellfun(@(x) x(1), skip_criteria);
%%%%%%%%%%%%%%% Set Parameters %%%%%%%%%%%%
dwot_get_default_params;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.template_initialization_mode = 0; 
param.nms_threshold = 0.4;
param.model_paths = model_paths;
param.b_calibrate = true;
param.n_calibration_images = 100;
param.detection_threshold = 0;

if exist('renderer','var')
  renderer.delete();
  clear renderer;
end

% Initialize renderer
if ~isfield(param,'renderer')
  renderer = Renderer();
  if ~renderer.initialize(model_paths, 700, 700, 0, 0, 0, 0, 25)
    error('fail to load model');
  end
end

% detector name
[ detector_model_name ] = dwot_get_detector_name(CLASS, SUB_CLASS, model_names, param);
detector_name = sprintf('%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d',...
    LOWER_CASE_CLASS,  detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs));

detector_file_name = sprintf('%s.mat', detector_name);

detection_result_file = sprintf('%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_sbin_%d_nms_%0.2f_skp_%s.txt',...
      DATA_SET, LOWER_CASE_CLASS, TYPE, detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs), sbin, param.nms_threshold, skip_name);


if exist(detector_file_name,'file')
  load(detector_file_name);
else
  [detectors] = dwot_make_detectors_grid(renderer, azs, els, yaws, fovs, 1:length(model_names), LOWER_CASE_CLASS, param, visualize_detector);
  [detectors, detector_table]= dwot_make_table_from_detectors(detectors);
  if sum(cellfun(@(x) isempty(x), detectors))
    error('Detector Not Completed');
  end
  eval(sprintf(['save -v7.3 ' detector_file_name ' detectors detector_table']));
  % detectors = dwot_make_detectors(renderer, azs, els, yaws, fovs, param, visualize_detector);
  % eval(sprintf(['save ' detector_name ' detectors']));
end

if param.b_calibrate
  calibrated_detector_file_name = sprintf('%s_cal.mat', detector_name);
  if exist(calibrated_detector_file_name,'file')
    load(calibrated_detector_file_name);
  else
    detectors = dwot_calibrate_detectors(detectors, LOWER_CASE_CLASS, VOCopts, param);
    eval(sprintf(['save -v7.3 ' calibrated_detector_file_name ' detectors']));
  end
  param.detectors = detectors;
end

%%%%% For Debuggin purpose only
param.detectors              = detectors;
param.detect_pyramid_padding = 10;
%%%%%%%%%%%%
renderings = cellfun(@(x) x.rendering_image, detectors, 'UniformOutput', false);

if COMPUTING_MODE == 0
  % for CPU convolution, use fconvblas which handles template inversion
  templates_cpu = cellfun(@(x) single(x.whow), detectors,'UniformOutput',false);
elseif COMPUTING_MODE == 1
  % for GPU convolution, invert template
  templates_gpu = cellfun(@(x) gpuArray(single(x.whow(end:-1:1,end:-1:1,:))), detectors,'UniformOutput',false);
elseif COMPUTING_MODE == 2
  templates_gpu = cellfun(@(x) gpuArray(single(x.whow(end:-1:1,end:-1:1,:))), detectors,'UniformOutput',false);
  templates_cpu = cellfun(@(x) single(x.whow), detectors,'UniformOutput',false);
else
  error('Computing mode undefined');
end


% load dataset
[gtids,t] = textread(sprintf(VOCopts.imgsetpath,[LOWER_CASE_CLASS '_' TYPE]),'%s %d');

N_IMAGE = length(gtids);
% N_IMAGE = 1500;
% extract ground truth objects
npos = 0;
tp = cell(1,N_IMAGE);
fp = cell(1,N_IMAGE);
% atp = cell(1,N_IMAGE);
% afp = cell(1,N_IMAGE);
detScore = cell(1,N_IMAGE);
detectorId = cell(1,N_IMAGE);
detIdx = 0;

clear gt;
gt(length(gtids))=struct('BB',[],'diff',[],'det',[]);

% Make empty detection save file
detection_result_file = dwot_save_detection([], 'Result', detection_result_file, [], true);

for imgIdx=1:N_IMAGE
    fprintf('%d/%d ',imgIdx,N_IMAGE);
    imgTic = tic;
    % read annotation
    recs(imgIdx)=PASreadrecord(sprintf(VOCopts.annopath,gtids{imgIdx}));
    
    clsinds = strmatch(LOWER_CASE_CLASS,{recs(imgIdx).objects(:).class},'exact');

    if dwot_skip_criteria(recs(imgIdx).objects(clsinds), skip_criteria); continue; end

    gt(imgIdx).BB=cat(1,recs(imgIdx).objects(clsinds).bbox)';
    gt(imgIdx).diff=[recs(imgIdx).objects(clsinds).difficult];
    gt(imgIdx).det=false(length(clsinds),1);
    
    
    im = imread([VOCopts.datadir, recs(imgIdx).imgname]);
    imSz = size(im);
    if COMPUTING_MODE == 0
      [bbsAllLevel, hog, scales] = dwot_detect( im, templates_cpu, param);
%       [hog_region_pyramid, im_region] = dwot_extract_region_conv(im, hog, scales, bbsNMS, param);
%       [bbsNMS_MCMC] = dwot_mcmc_proposal_region(im, hog, scale, hog_region_pyramid, param);
    elseif COMPUTING_MODE == 1
      % [bbsNMS ] = dwot_detect_gpu_and_cpu( im, templates, templates_cpu, param);
      [bbsAllLevel, hog, scales] = dwot_detect_gpu( im, templates_gpu, param);
%       [hog_region_pyramid, im_region] = dwot_extract_region_fft(im, hog, scales, bbsNMS, param);
    elseif COMPUTING_MODE == 2
      [bbsAllLevel, hog, scales] = dwot_detect_combined( im, templates_gpu, templates_cpu, param);
    else
      error('Computing Mode Undefined');
    end
    fprintf(' time to convolution: %0.4f\n', toc(imgTic));
    
    
    % Automatically sort them according to the score and apply NMS
    bbsNMS = esvm_nms(bbsAllLevel, param.nms_threshold);
    
    bbsNMS_clip = clip_to_image(bbsNMS, [1 1 imSz(2) imSz(1)]);
    [bbsNMS_clip, tp{imgIdx}, fp{imgIdx}, detScore{imgIdx}, ~] = dwot_compute_positives(bbsNMS_clip, gt(imgIdx), param);
    
    [~, img_file_name] = fileparts(recs(imgIdx).imgname);
    dwot_save_detection(bbsNMS_clip, 'Result', detection_result_file, img_file_name);
    
    if visualize_detection && ~isempty(clsinds)
      % figure(2);
      nDet = size(bbsNMS,1);
      if nDet > 0
        bbsNMS(:,9) = bbsNMS_clip(:,9);
      end
      
      tpIdx = logical(tp{imgIdx});
      % tpIdx = bbsNMS(:, 9) > param.min_overlap;
      
      % Original images
      subplot(221);
      imagesc(im); axis off; axis equal;
      
      % True positives
      subplot(222);
      dwot_draw_overlap_detection(im, bbsNMS(tpIdx,:), renderings, 1, 50, visualize_detection, [0.3, 0.7, 0] );
      
      subplot(223);
      dwot_draw_overlap_detection(im, bbsNMS(tpIdx,:), renderings, inf, 50, visualize_detection, [0.3, 0.5, 0.2] );
      
      % False positives
      subplot(224);
      dwot_draw_overlap_detection(im, bbsNMS(~tp{imgIdx},:), renderings, 4, 50, visualize_detection, [0.3, 0.7, 0]);
      
      drawnow;
      spaceplots();
      
      drawnow;
      save_name = sprintf('%s_%s_%s_%s_cal_%d_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_sbin_%d_nms_%0.2f_imgIdx_%d.jpg',...
        DATA_SET, LOWER_CASE_CLASS, TYPE, detector_model_name, param.b_calibrate,  n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs), sbin, param.nms_threshold, imgIdx);
      print('-djpeg','-r150',['Result/' LOWER_CASE_CLASS '_' TYPE '/' save_name]);
      
      %  waitforbuttonpress;
    end
      
    npos=npos+sum(~gt(imgIdx).diff);
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

close all;
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
save_name = sprintf('AP_%s_%s_%s_cal_%d_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_sbin_%d_nms_%0.2f_skp_%s_N_IM_%d.png',...
        LOWER_CASE_CLASS, TYPE, detector_model_name, param.b_calibrate, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs), sbin, param.nms_threshold, skip_name, N_IMAGE);

print('-dpng','-r150',['Result/' LOWER_CASE_CLASS '_' TYPE '/' save_name])
