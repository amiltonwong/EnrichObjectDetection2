addpath('HoG');
addpath('HoG/features');
addpath('Util');
addpath('DecorrelateFeature/');
addpath('../MatlabRenderer/');
addpath('../MatlabRenderer/bin');
addpath('../MatlabCUDAConv/');
addpath('3rdParty/SpacePlot');
addpath('3rdParty/MinMaxSelection');
addpath('Diagnosis');

% Addpath doesn't work pwd in the code uses current directory so move to
% the directory.
DATA_SET = 'PASCAL';
dwot_set_datapath;

COMPUTING_MODE = 1;
CLASS = 'Car';
% CLASS = 'Bicycle';
SUB_CLASS = [];
LOWER_CASE_CLASS = lower(CLASS);
TEST_TYPE = 'val';
mkdir('Result',[LOWER_CASE_CLASS '_GT_' TEST_TYPE]);

if COMPUTING_MODE > 0
  gdevice = gpuDevice(1);
  reset(gdevice);
  cos(gpuArray(1));
end
daz = 45;
del = 20;
dfov = 10;
dyaw = 10;

azs = 0:7.5:352.5; % azs = [azs , azs - 10, azs + 10];
els = 0:10:20;
fovs = [25 50];
yaws = 0;
n_cell_limit = [300];
lambda = [0.015];
detection_threshold = 80;

visualize_detection = true;
visualize_detector = false;

sbin = 6;
n_level = 10;
n_proposals = 1;

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

% skip_criteria = {'empty', 'truncated','difficult','occluded'};
skip_criteria = {'empty', 'truncated','difficult'};
% skip_criteria = {'empty'};
skip_name = cellfun(@(x) x(1), skip_criteria);

%%%%%%%%%%%%%%% Set Parameters %%%%%%%%%%%%
dwot_get_default_params;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.template_initialization_mode = 0; 
param.nms_threshold = 0.4;
param.model_paths = model_paths;


param.b_calibrate = 0;
param.n_calibration_images = 20;
param.calibration_mode = 'gaussian';

param.detection_threshold = 100;
param.image_scale_factor = 2;



% MCMC Setting
param.mcmc_max_iter = 50;
param.extraction_padding_ratio = 0.2;
param.region_extraction_levels = 1;

color_range = [-inf 100:20:300 inf];

% param.gather_

% detector name
[ detector_model_name ] = dwot_get_detector_name(CLASS, SUB_CLASS, model_names, param);
detector_name = sprintf('%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d',...
    LOWER_CASE_CLASS,  detector_model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs));

detector_file_name = sprintf('%s.mat', detector_name);

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


%% Make Detectors
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


%% Make templates, these are just pointers to the templates in the detectors,
% The following code copies variables to GPU or make pointers to memory
% according to the computing mode.
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

template_size = cell2mat(cellfun(@(x) size(x)', templates_gpu ,'UniformOutput',false));
max_template_size = max(template_size,[],2);


%% Set variables for detection
[gtids,t] = textread(sprintf(VOCopts.imgsetpath,[LOWER_CASE_CLASS '_' TEST_TYPE]),'%s %d');

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
for imgIdx=1:N_IMAGE
    fprintf('%d/%d ', imgIdx, N_IMAGE);
    imgTic = tic;
    % read annotation
    recs(imgIdx)=PASreadrecord(sprintf(VOCopts.annopath,gtids{imgIdx}));
    
    clsinds = strmatch(LOWER_CASE_CLASS,{recs(imgIdx).objects(:).class},'exact');

    if dwot_skip_criteria(recs(imgIdx).objects(clsinds), skip_criteria); continue; end

    gt(imgIdx).BB = param.image_scale_factor * cat(1, recs(imgIdx).objects(clsinds).bbox)';
    gt(imgIdx).diff = [recs(imgIdx).objects(clsinds).difficult];
    gt(imgIdx).det = zeros(length(clsinds),1);
    
    im = imread([VOCopts.datadir, recs(imgIdx).imgname]);
    im = imresize(im,param.image_scale_factor);
    imSz = size(im);

    for i_object = 1 : numel(clsinds)
      BB = gt(imgIdx).BB(:,i_object)';
      sbin = param.sbin;
      
      clip_padded_bbox = dwot_clip_pad_bbox(BB, param.extraction_padding_ratio, imSz);
      width = BB(3)-BB(1);
      height = BB(4)-BB(2);
      gt_pad_im              = im(clip_padded_bbox(2):clip_padded_bbox(4), clip_padded_bbox(1):clip_padded_bbox(3),:);
      gt_size                  = size(gt_pad_im);
      gt_start                 = [BB(1)-clip_padded_bbox(1)+1, BB(2)-clip_padded_bbox(2)+1];
      crop_BB                = [ gt_start(1), gt_start(2), gt_start(1)+width, gt_start(2)+height ];
      
      temp_gt.BB = crop_BB';
      temp_gt.diff = gt(imgIdx).diff(i_object);
      temp_gt.det = 0;
      
      search_scale = max_template_size(1:2)' * sbin ./ [height width];
      if max(search_scale) >= 1
          sbin = 4;
          search_scale = max_template_size(1:2)' * sbin ./ [height width];
      end
      
      param.detect_min_scale             = min(1, min(search_scale) / 1.4);
      param.detect_max_scale            = min(1, min(search_scale) *1.2);
      param.detect_levels_per_octave = 50;
      
        if COMPUTING_MODE == 0
            [bbsAllLevel, hog, scales] = dwot_detect( gt_pad_im, templates_cpu, param);
        elseif COMPUTING_MODE == 1
            [bbsAllLevel, hog, scales] = dwot_detect_gpu( gt_pad_im, templates_gpu, param);
        elseif COMPUTING_MODE == 2
            [bbsAllLevel, hog, scales] = dwot_detect_combined( gt_pad_im, templates_gpu, templates_cpu, param);
        else
            error('Computing Mode Undefined');
        end
      
        bbsNMS = esvm_nms(bbsAllLevel, param.nms_threshold);
        n_mcmc = min(n_proposals, size(bbsNMS,1));

        % dwot_draw_overlap_detection(gt_region, bbsNMS, renderings, 3, 50, visualize_detection, [0.3, 0.7, 0] , color_range );
        [hog_region_pyramid, im_region] = dwot_extract_hog(hog, scales, detectors, bbsNMS(1:n_mcmc,:), param, gt_pad_im);


        figure(2);  
        % [best_proposals] = dwot_mcmc_proposal_region(renderer, hog_region_pyramid, im_region, detectors, param, gt_pad_im, true);
        % [best_proposals] = dwot_breadth_first_search_proposal_region(hog_region_pyramid, im_region, detectors, detector_table, param, im);
        [best_proposals, detectors, detector_table] = dwot_binary_search_proposal_region(hog_region_pyramid, im_region, detectors, detector_table, renderer, param, gt_pad_im);
        
        
        
        figure(1);
        for proposal_idx = 1:n_mcmc
              bbsNMS_clip = clip_to_image(bbsNMS, [1 1 gt_size(2) gt_size(1)]);
              [bbsNMS_clip, tp{imgIdx}, fp{imgIdx}, ~, ~] = dwot_compute_positives(bbsNMS_clip, temp_gt, param);
              nDet = size(bbsNMS,1);
              if nDet > 0
                bbsNMS(:,9) = bbsNMS_clip(:,9);
              end
              
              subplot(131);
            imagesc(gt_pad_im);
            rectangle('position',crop_BB - [0 0 crop_BB(1:2)],'edgecolor',[0.5 0.5 0.5],'LineWidth',3);
            axis equal; axis tight;
            
            subplot(132);
            dwot_draw_overlap_detection(gt_pad_im, bbsNMS, renderings, n_proposals, 50, visualize_detection, [0.3, 0.7, 0] , color_range );
            axis equal; axis tight;
                        
            subplot(133);
            bbox = best_proposals{proposal_idx}.image_bbox;
                        
            bbox(12) = best_proposals{proposal_idx}.score;
            bbsNMS_clip = clip_to_image(bbox, [1 1 gt_size(2) gt_size(1)]);
            [bbsNMS_clip, tp{imgIdx}, fp{imgIdx}, ~, ~] = dwot_compute_positives(bbsNMS_clip, temp_gt, param);
            bbox(:,9) = bbsNMS_clip(:,9);
            
            dwot_draw_overlap_detection(gt_pad_im, bbox, best_proposals{proposal_idx}.rendering_image, n_mcmc, 50, true,  [0.3, 0.7, 0] , color_range);
            axis equal; axis tight;

            spaceplots();
          
%           fprintf('press any button to continue\n');
%           waitforbuttonpress
            save_name = sprintf(['%s_%s_%s_GT_MCMC_%s_cal_%d_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d',...
                      '_mcmc_%d_imgIdx_%d_obj_%d.jpg'],...
                      DATA_SET, LOWER_CASE_CLASS, TEST_TYPE, detector_model_name, param.b_calibrate,  n_cell_limit, lambda, ...
                      numel(azs), numel(els), numel(yaws), numel(fovs), param.mcmc_max_iter ,imgIdx, i_object);
            print('-djpeg','-r150',['Result/' LOWER_CASE_CLASS '_GT_' TEST_TYPE '/' save_name]);
        end
    end
   fprintf(' convolution time: %0.4f\n', toc(imgTic));
end
