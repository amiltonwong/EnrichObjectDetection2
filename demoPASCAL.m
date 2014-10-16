addpath('HoG');
addpath('HoG/features');
addpath('Util');
addpath('DecorrelateFeature/');
addpath('../MatlabRenderer/');
addpath('../MatlabRenderer/bin');
addpath('../MatlabCUDAConv/');
addpath('3rdParty/SpacePlot');
addpath('3rdParty/MinMaxSelection');
% addpath('Diagnosis');

DATA_SET = 'PASCAL';
dwot_set_datapath;

COMPUTING_MODE = 1;
CLASS = 'Car';
SUB_CLASS = [];     % Sub folders
LOWER_CASE_CLASS = lower(CLASS);
TEST_TYPE = 'val';
SAVE_PATH = fullfile('Result',[LOWER_CASE_CLASS '_' TEST_TYPE]);
if ~exist(SAVE_PATH,'dir'); mkdir(SAVE_PATH); end

DEVICE_ID = 0; % 0-base indexing

if COMPUTING_MODE > 0
  gdevice = gpuDevice(DEVICE_ID + 1); % Matlab use 1 base indexing
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

visualize_detection = true;
visualize_detector = false;

sbin = 6;
n_level = 15;
n_max_proposals = 10;

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
              'Porsche_911',...
              '2009 Toyota Cargo'};

use_idx = ismember(model_names,models_to_use);

model_names = model_names(use_idx);
model_paths = model_paths(use_idx);

% skip_criteria = {'empty', 'truncated','difficult'};
skip_criteria = {'empty'};
skip_name = cellfun(@(x) x(1), skip_criteria);

%%%%%%%%%%%%%%% Set Parameters %%%%%%%%%%%%
dwot_get_default_params;

param.template_initialization_mode = 0; 
param.nms_threshold = 0.4;
param.model_paths = model_paths;

param.b_calibrate = 0;      % apply callibration if > 0
param.n_calibration_images = 100; 
param.calibration_mode = 'gaussian';

param.detection_threshold = 80;
param.image_scale_factor = 2; % scale image accordingly and detect on the scaled image

% Tuning mode == 0, no tuning
%             == 1, MCMC
%             == 2, Breadth first search
%             == 3, Quasi-Newton method (BFGS)
param.proposal_tuning_mode = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param.color_range = [-inf 120:10:300 inf];


% detector name
[ detector_model_name ] = dwot_get_detector_name(CLASS, SUB_CLASS, model_names, param);
detector_name = sprintf('%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d',...
        LOWER_CASE_CLASS,  detector_model_name, n_cell_limit, lambda,...
        numel(azs), numel(els), numel(yaws), numel(fovs));

detector_file_name = sprintf('%s.mat', detector_name);

%% Make empty detection save file
detection_result_file = sprintf(['%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_scale_',...
        '%0.2f_sbin_%d_level_%d_nms_%0.2f_skp_%s_server_%s.txt'],...
        DATA_SET, LOWER_CASE_CLASS, TEST_TYPE, detector_model_name, n_cell_limit, lambda,...
        numel(azs), numel(els), numel(yaws), numel(fovs), param.image_scale_factor, sbin,...
        n_level, param.nms_threshold, skip_name, server_id.num);

% Check duplicate file name and return different name
detection_result_file = dwot_save_detection([], SAVE_PATH, detection_result_file, [], true);

if param.proposal_tuning_mode > 0
    detection_tuning_result_file = sprintf(['%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_scale_',...
        '%0.2f_sbin_%d_level_%d_nms_%0.2f_skp_%s_server_%s_tuning.txt'],...
        DATA_SET, LOWER_CASE_CLASS, TEST_TYPE, detector_model_name, n_cell_limit, lambda,...
        numel(azs), numel(els), numel(yaws), numel(fovs), param.image_scale_factor, sbin,...
        n_level, param.nms_threshold, skip_name, server_id.num);
    % Check duplicate file name and return different name
    detection_tuning_result_file = dwot_save_detection([], SAVE_PATH, detection_tuning_result_file, [], true);
end

fprintf('\nThe result will be saved on %s\n',detection_result_file);

%% Make Renderer
if ~exist(detector_file_name,'file')  || param.proposal_tuning_mode > 0
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

%% Set variables for detection
% ground truth is required to plot figure.
[gtids,t] = textread(sprintf(VOCopts.imgsetpath,[LOWER_CASE_CLASS '_' TEST_TYPE]),'%s %d');

N_IMAGE = length(gtids);

clear gt;
gt = struct('BB',[],'diff',[],'det',[]);

for imgIdx=1:N_IMAGE
    fprintf('%d/%d ',imgIdx,N_IMAGE);
    imgTic = tic;
    % read annotation
    recs=PASreadrecord(sprintf(VOCopts.annopath,gtids{imgIdx}));
    
    clsinds = strmatch(LOWER_CASE_CLASS,{recs.objects(:).class},'exact');

    if dwot_skip_criteria(recs.objects(clsinds), skip_criteria); continue; end

    gt.BB = param.image_scale_factor * cat(1, recs.objects(clsinds).bbox)';
    gt.diff = [recs.objects(clsinds).difficult];
    gt.det = zeros(length(clsinds),1);
    
    im = imread([VOCopts.datadir, recs.imgname]);
    im = imresize(im, param.image_scale_factor);
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
    fprintf('convolution time: %0.4f\n', toc(imgTic));
    
    % Automatically sort them according to the score and apply NMS
    bbsNMS = esvm_nms(bbsAllLevel, param.nms_threshold);
    
    bbsNMS_clip = clip_to_image(bbsNMS, [1 1 imSz(2) imSz(1)]);
    [bbsNMS_clip, tp ] = dwot_compute_positives(bbsNMS_clip, gt, param);
    if size(bbsNMS,1) > 0
        bbsNMS(:,9) = bbsNMS_clip(:,9);
    end
    [~, img_file_name] = fileparts(recs.imgname);
    dwot_save_detection(esvm_nms(bbsAllLevel, 0.7), SAVE_PATH, detection_result_file, ...
                                 img_file_name, false, 1); % save mode != 0 to save template index
    
    if visualize_detection && ~isempty(clsinds)
        tpIdx = logical(tp); % Index of bounding boxes that will be printed as ground truth
        dwot_visualize_result;
        save_name = sprintf(['%s_%s_%s_%s_cal_%d_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_',...
                            'scale_%0.2f_sbin_%d_level_%d_nms_%0.2f_imgIdx_%d.jpg'],...
                            DATA_SET, LOWER_CASE_CLASS, TEST_TYPE, detector_model_name,...
                            param.b_calibrate,  n_cell_limit, lambda, numel(azs), numel(els),...
                            numel(yaws), numel(fovs), param.image_scale_factor, sbin, n_level,...
                            param.nms_threshold, imgIdx);
        print('-djpeg','-r150',fullfile(SAVE_PATH, save_name));
    end
    
    
    %% Proposal Tuning
    if param.proposal_tuning_mode > 0
        tuningTic = tic;

        n_proposals = min(n_max_proposals, size(bbsNMS,1));

        [hog_region_pyramid, im_region] = dwot_extract_hog(hog, scales, detectors, ...
                              bbsNMS(1:n_proposals,:), param, im);
        
        switch param.proposal_tuning_mode
            case 1
                [best_proposals] = dwot_mcmc_proposal_region(renderer, hog_region_pyramid, im_region,...
                                                detectors, param, im, false);
            case 2
                [best_proposals] = dwot_breadth_first_search_proposal_region(hog_region_pyramid, ...
                                                im_region, detectors, detector_table, param, im);
                % [best_proposals, detectors, detector_table] = dwot_binary_search_proposal_region(...
                %               hog_region_pyramid, im_region, detectors, detector_table, renderer, param, im);
            case 3
                [best_proposals] = dwot_bfgs_proposal_region(renderer, hog_region_pyramid, im_region,...
                                                detectors, detector_table, param, im);
            otherwise
                error('Undefined tuning mode');
        end
        fprintf(' tuning time : %0.4f\n', toc(tuningTic));
        
        bbsProposal = zeros(n_proposals,12);
        % For each proposals draw original, before and after
        for proposal_idx = 1:n_proposals
            % fill out the box proposal infomation
            bbsProposal(proposal_idx, 1:4) = best_proposals{proposal_idx}.image_bbox;
            bb_clip = clip_to_image(bbsProposal(proposal_idx, :), [1 1 imSz(2) imSz(1)]);
            bb_clip = dwot_compute_positives(bb_clip, gt, param);
            bbsProposal(proposal_idx, 9) = bb_clip(9);
            bbsProposal(proposal_idx, 12) = best_proposals{proposal_idx}.score;
            bbsProposal(proposal_idx, 11) = 1;
            dwot_visualize_proposal_tuning(bbsNMS(proposal_idx,:), bbsProposal(proposal_idx,:), ...
                                            best_proposals{proposal_idx}, im, detectors, param);
            save_name = sprintf(['%s_%s_%s_tuning_%s_%s_cal_%d_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d',...
                      '_mcmc_%d_imgIdx_%d_obj_%d.jpg'],...
                      DATA_SET, LOWER_CASE_CLASS, TEST_TYPE, detector_model_name, param.b_calibrate,...
                      n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs), ...
                      param.mcmc_max_iter ,imgIdx, proposal_idx);
            print('-djpeg','-r150',fullfile(SAVE_PATH, save_name));
        end
    end
    bbsNMS(1:n_proposals,:) = bbsProposal;
    dwot_save_detection(bbsNMS, SAVE_PATH, detection_tuning_result_file, ...
                                 img_file_name, false, 1); % save mode != 0 to save template index
end

%% Vary NMS threshold
nms_thresholds = 0.2 : 0.05 : 0.7;
ap = zeros(numel(nms_thresholds),1);
ap_save_names = cell(numel(nms_thresholds),1);
for i = 1:numel(nms_thresholds)
    nms_threshold = nms_thresholds(i);
    ap(i) = dwot_analyze_and_visualize_pascal_results(fullfile('Result',detection_result_file), ...
                        detectors, [], VOCopts, param, skip_criteria, param.color_range, ...
                        nms_threshold, false);
                        

    ap_save_names{i} = sprintf(['AP_%s_%s_%s_%s_cal_%d_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_',...
                        '%d_scale_%0.2f_sbin_%d_level_%d_nms_%0.2f_skp_%s_N_IM_%d_%s.png'],...
                        DATA_SET, LOWER_CASE_CLASS, TEST_TYPE, detector_model_name,...
                        param.b_calibrate, n_cell_limit, lambda, numel(azs), numel(els),...
                        numel(yaws), numel(fovs), param.image_scale_factor, sbin, n_level,...
                        nms_threshold, skip_name, N_IMAGE, server_id.num);

     print('-dpng','-r150',fullfile(SAVE_PATH, ap_save_names{i}));
end

if param.proposal_tuning_mode > 0
    ap_tuning = dwot_analyze_and_visualize_pascal_results(fullfile(SAVE_PATH,...
                        detection_tuning_result_file), detectors, [], VOCopts, param,...
                        skip_criteria, param.color_range, param.nms_threshold, false);
                        
    ap_tuning_save_name = sprintf(['AP_%s_%s_%s_%s_cal_%d_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_',...
                        '%d_scale_%0.2f_sbin_%d_level_%d_nms_%0.2f_skp_%s_N_IM_%d_%s_tuning.png'],...
                        DATA_SET, LOWER_CASE_CLASS, TEST_TYPE, detector_model_name,...
                        param.b_calibrate, n_cell_limit, lambda, numel(azs), numel(els),...
                        numel(yaws), numel(fovs), param.image_scale_factor, sbin, n_level,...
                        nms_threshold, skip_name, N_IMAGE, server_id.num);

     print('-dpng','-r150',fullfile(SAVE_PATH, ap_tuning_save_name));
end

if ~isempty(server_id)
    for i = 1:numel(nms_thresholds)
        system(['scp ', fullfile(SAVE_PATH, ap_save_names{i}),...
            ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/Result/',...
            LOWER_CASE_CLASS '_' TEST_TYPE]);
    end
    
    if param.proposal_tuning_mode > 1
        system(['scp ', fullfile(SAVE_PATH, ap_tuning_save_name),...
            ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/Result/',...
            LOWER_CASE_CLASS '_' TEST_TYPE]);
        system(['scp ' fullfile(SAVE_PATH, detection_tuning_result_file),...
            ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/Result/']);
    end
    system(['scp ' fullfile(SAVE_PATH, detection_result_file),...
        ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/Result/']);
end