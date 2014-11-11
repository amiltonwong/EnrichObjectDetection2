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

DATA_SET = '3DObject';
dwot_set_datapath;

% Computing Mode  = 0, CPU
%                 = 1, GPU
%                 = 2, Combined
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

azs = 0:15:345; 
els = 0:10:30;
fovs = [25];
yaws = 0;
n_cell_limit = [250];
lambda = [0.15];
detection_threshold = 40;

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

% models_to_use = {'atx_bike',...
%               'atx_bike_rot',...
%               'bmx_bike_4',...
%               'brooklyn_machine_works_bike',...
%               'downhill_bike',...
%               'glx_bike',...
%               'road_bike',...
%               'road_bike_rot'};

% models_to_use = {'2012-VW-beetle-turbo',...
%               'Kia_Spectra5_2006',...
%               '2008-Jeep-Cherokee',...
%               'Ford Ranger Updated',...
%               'BMW_X1_2013',...
%               'Honda_Accord_Coupe_2009',...
%               'Porsche_911',...
%               '2009 Toyota Cargo'};

% models_to_use = {'2012-VW-beetle-turbo',...
%             'Peugeot-207',...
%             'Scion_xB',...
%             '2007-Nissan-Versa-_-Tiida-SL',...
%             '2010-Chevrolet-Aveo5-LT',...
%             '2012-Citroen-DS4',...
%             'Kia_Spectra5_2006',...
%             'Opel-Meriva',...
%             'NM07',...
%             'Portugal_Racing_Junior',...
%             '2008-Jeep-Cherokee',...
%             '2012-Mercedes-Benz-M-Class',...
%             'BMW_X1_2013',...
%             'Benz_SL500_roofdown_2013',...
%             'Chevrolet Camaro',...
%             'Honda-Accord-3',...
%             'Honda_Accord_Coupe_2009',...
%             'Mercedes-Benz-C63-2012',...
%             'Nissan-Maxima_2009',...
%             'Porsche_911_wing_2014',...
%             '2001-2004 Ford Ranger Edge',...
%             '2013-Ford-F150-Eco-Boost-King-Ranch-4X4-Crew-Cab',...
%             'Ford Ranger Updated',...
%             '2002 Dodge Ram FedEx',...
%             '2009 Toyota Cargo',...
%             'Maserati-3500GT',...
%             'Skylark_Cruiser_1971'};

models_to_use = {'Honda-Accord-3'};

use_idx = ismember(model_names,models_to_use);

model_names = model_names(use_idx);
model_paths = model_paths(use_idx);

% skip_criteria = {'empty', 'truncated','difficult'};
skip_criteria = {'none'};
skip_name = cellfun(@(x) x(1), skip_criteria); % get the first character of the criteria

%%%%%%%%%%%%%%% Set Parameters %%%%%%%%%%%%
dwot_get_default_params;

param.template_initialization_mode = 0; 
param.nms_threshold = 0.4;
param.model_paths = model_paths;

param.b_calibrate = 0;      % apply callibration if > 0
param.n_calibration_images = 100; 
param.calibration_mode = 'gaussian';

param.detection_threshold = 40;
param.image_scale_factor = 2; % scale image accordingly and detect on the scaled image

% Tuning mode == 0, no tuning
%             == 1, MCMC
%             == 2, Breadth first search
%             == 3, Quasi-Newton method (BFGS)
param.proposal_tuning_mode = 'mcmc';


% Detection mode == 'dwot' ours
%                == 'cnn'
%                == 'dpm'
param.detection_mode = 'dwot';


% Save mode = 'd' detections
%           = 'dt' detection and template index
%           = 'dv' detection and viewpoint
%           = 'dtp' detection template index and proposal
%           = 'dvp' detection viewpoint and proposal
param.detection_save_mode = 'dt';
param.tuning_save_mode = 'dvp';
param.export_figure = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param.color_range = [-inf 40:10:120 inf];


% detector name
[ detector_model_name ] = dwot_get_detector_model_name( model_names, param);
[detector_name, detector_file_name] = dwot_get_detector_name( model_names, param);

%% Make empty detection save file
[detection_result_file, detection_result_common_name, file_temp_idx] = ...
    dwot_get_detection_result_file_name(DATA_SET, TEST_TYPE, SAVE_PATH, model_names,...
        param.detection_save_mode, server_id, param, '', true);
fprintf('\n%s\n',detection_result_file);

if ~strcmp(param.proposal_tuning_mode,'none')
    [detection_tuning_result_file, detection_tuning_result_common_name] = ...
        dwot_get_detection_result_file_name(DATA_SET, TEST_TYPE, SAVE_PATH, model_names,...
            param.tuning_save_mode, server_id, param, 'tuning', true, file_temp_idx);
    fprintf('%s\n',detection_tuning_result_file);
end

%% Make Renderer
if ~strcmp(param.proposal_tuning_mode, 'none') || ~exist('renderer','var') || ~exist(detector_file_name,'file')
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
    eval(sprintf(['save ' detector_file_name ' detectors detector_table']));
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
    if isempty(server_id) || ~strcmp(server_id.num,'capri7')
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
% [gtids,t] = textread(sprintf(VOCopts.imgsetpath,[LOWER_CASE_CLASS '_' TEST_TYPE]),'%s %d');
[gt, image_path] = dwot_3d_object_dataset(DATA_PATH, CLASS);
N_IMAGE = length(gt)/4;
% N_IMAGE = 20;

n_views = 8;
max_azimuth_difference = 360/n_views/2;

if strcmp(CLASS, 'Bicycle')
    prediction_azimuth_offset = 90;
    prediction_azimuth_rotation_direction = -1;
else
    prediction_azimuth_offset = 180;
    prediction_azimuth_rotation_direction = 1;
end

detection_time_per_image = zeros(N_IMAGE,1);
tuning_time_per_image = zeros(N_IMAGE,1);

for image_idx=1:N_IMAGE
    fprintf('%d/%d ',image_idx,N_IMAGE);
    % read annotation
    % recs = PASreadrecord(sprintf(VOCopts.annopath,gtids{img_idx}));
    % clsinds = find(strcmp(LOWER_CASE_CLASS,{recs.objects(:).class}));
    % [b_skip, skip_object_idx] = dwot_skip_criteria(recs.objects(clsinds), skip_criteria); 
    % if b_skip; continue; end;
    % clsinds = clsinds(skip_object_idx);
    % ground_truth_bounding_boxes = param.image_scale_factor * cat(1, recs.objects(clsinds).bbox);
    
    
    ground_truth_bounding_boxes =  param.image_scale_factor * gt{image_idx}.BB';
    ground_truth_azimuth = gt{image_idx}.azimuth;    

    
    % im   = imread([VOCopts.datadir, recs.imgname]);
    im = imread([image_path{image_idx}]);

    img_file_name = regexp(image_path{image_idx}, ['\/(' LOWER_CASE_CLASS '_\d+\/\w+)\.'],'tokens');
    img_file_name = img_file_name{1}{1};

    im   = imresize(im, param.image_scale_factor);
    im_size = size(im);
    
    imgTic = tic;
    
    if COMPUTING_MODE == 0
      [bbsAllLevel, hog, scales] = dwot_detect( im, templates_cpu, param);
    elseif COMPUTING_MODE == 1
      [bbsAllLevel, hog, scales] = dwot_detect_gpu( im, templates_gpu, param);
    end
    detection_time_per_image(image_idx) = toc(imgTic);
    fprintf('convolution time: %0.4f\n', detection_time_per_image(image_idx));
    
    % Automatically sort them according to the score and apply NMS
    bbsAllLevel = dwot_return_null_padded_box(bbsAllLevel, [], 12)
    proposal_formatted_bounding_boxes = esvm_nms(bbsAllLevel, param.nms_threshold);
    prediction_azimuth = proposal_formatted_bounding_boxes(:,10);
    % [~, img_file_name] = fileparts(recs.imgname);

    save_structure = dwot_formatted_bounding_boxes_to_save_structure(esvm_nms(bbsAllLevel, 0.7));
    dwot_save_detection(SAVE_PATH, detection_result_file, false, ...
        param.detection_save_mode, save_structure, img_file_name); % save mode != 0 to save template index
    
    if visualize_detection 
        dwot_visualize_predictions_in_quadrants(im, proposal_formatted_bounding_boxes,...
                            ground_truth_bounding_boxes, detectors, param);
        if param.export_figure
            save_name = sprintf(['%s_img_%d.jpg'], detection_result_common_name, image_idx);
            print('-djpeg','-r150',fullfile(SAVE_PATH, save_name));
        end
    end
    
    
    %% Proposal Tuning
    if ~strcmp(param.proposal_tuning_mode,'none')


        n_proposal = min(n_max_proposals, size(proposal_formatted_bounding_boxes,1));
        [hog_region_pyramids, im_regions] = dwot_extract_hog(hog, scales, detectors, ...
                              proposal_formatted_bounding_boxes(1:n_proposal,:), param, im);
                          
        tuningTic = tic;
        switch param.proposal_tuning_mode
            case 'mcmc'
                [best_proposals] = dwot_mcmc_proposal_region(renderer, hog_region_pyramids, im_regions,...
                                                detectors, param, im, false);
            case 'breadthfirst'
                [best_proposals] = dwot_breadth_first_search_proposal_region(hog_region_pyramids, ...
                                                im_regions, detectors, detector_table, param, im);
                % [best_proposals, detectors, detector_table] = dwot_binary_search_proposal_region(...
                %       hog_region_pyramid, im_region, detectors, detector_table, renderer, param, im);
            case 'bfgs'
                [best_proposals] = dwot_bfgs_proposal_region(renderer, hog_region_pyramids, im_regions,...
                                                detectors, detector_table, param, im);
            case 'dwot'
                if COMPUTING_MODE == 0
                  [bbsAllLevel, hog, scales] = dwot_detect( im, templates_cpu, param);
                elseif COMPUTING_MODE == 1
                  [bbsAllLevel, hog, scales] = dwot_detect_gpu( im, templates_gpu, param);
                end
            otherwise
               error('Undefined tuning mode');
        end
        tuning_time_per_image(image_idx) = toc(tuningTic);
        fprintf(' tuning time : %0.4f\n', tuning_time_per_image(image_idx));

        [tuned_prediction_boxes, tuned_prediction_scores, tuned_prediction_azimuth,...
            tuned_prediction_elevation, tuned_prediction_yaw, tuned_prediction_fov,...
            tuned_prediction_renderings, tuned_prediction_depths] = dwot_extract_proposals(best_proposals);

        tuned_formatted_bounding_boxes = dwot_predictions_to_formatted_bounding_boxes( tuned_prediction_boxes,...
                                               tuned_prediction_scores, [], [], tuned_prediction_azimuth);
 
        % For each proposals draw original, before and after
        if visualize_detection 
            [proposal_boxes, proposal_scores, proposal_template_indexes] = dwot_formatted_bounding_boxes_to_predictions(...
                                                             proposal_formatted_bounding_boxes(1:n_proposal,:));
            for proposal_idx = n_proposal:-1:1
                current_proposal_detector = detectors{proposal_template_indexes(proposal_idx)};

                dwot_visualize_proposal_tuning(im,...
                         proposal_boxes(proposal_idx, :), proposal_scores(proposal_idx),...
                         current_proposal_detector.rendering_image, current_proposal_detector.rendering_depth,...
                         tuned_prediction_boxes(proposal_idx,:), tuned_prediction_scores(proposal_idx),...
                         tuned_prediction_renderings{proposal_idx}, tuned_prediction_depths{proposal_idx},...
                         ground_truth_bounding_boxes, param);
                if param.export_figure
                    save_name = sprintf(['%s_tuning_%d_img_%d_obj_%d.jpg'],...
                          detection_tuning_result_common_name, param.proposal_tuning_mode,...
                          image_idx, proposal_idx);
                    print('-djpeg','-r150',fullfile(SAVE_PATH, save_name));
                end
            end
        end
        
        save_tuned_structure = dwot_formatted_bounding_boxes_to_save_structure(...
            tuned_formatted_bounding_boxes, proposal_formatted_bounding_boxes(1:n_proposal,:));
        dwot_save_detection(SAVE_PATH, detection_tuning_result_file, false, ...
            param.tuning_save_mode, save_tuned_structure, img_file_name); % save mode != 0 to save template index
    
%         dwot_save_detection(tuned_formatted_bounding_boxes, SAVE_PATH, detection_tuning_result_file, ...
%                                  img_file_name, false, 1); % save mode != 0 to save template index
    end
end

close all;  % space plot casues problem when using different subplot grid

%% Vary NMS threshold
nms_thresholds = 0.3:0.1:0.7;
ap = zeros(numel(nms_thresholds),1);
ap_dwot_cnn = zeros(numel(nms_thresholds),1);
ap_save_names = cell(numel(nms_thresholds),1);
ap_detection_save_names = cell(numel(nms_thresholds),1);

for i = 1:numel(nms_thresholds)
    nms_threshold = nms_thresholds(i);
%     ap(i) = dwot_analyze_and_visualize_cnn_results(fullfile(SAVE_PATH, detection_result_file), ...
%                         detectors, VOCopts, param, nms_threshold);
    ap(i) = dwot_analyze_and_visualize_3D_object_results( fullfile(SAVE_PATH, detection_result_file), ...
                            detectors, [], param, DATA_PATH, CLASS, param.color_range, nms_threshold, false, ...
                            prediction_azimuth_rotation_direction, prediction_azimuth_offset);

    ap_save_names{i} = sprintf(['AP_%s_nms_%0.2f.png'],...
                        detection_result_common_name, nms_threshold);

     print('-dpng','-r150',fullfile(SAVE_PATH, ap_save_names{i}));
end

% if ~exist('/scratch/chrischoy/Result/','dir'); mkdir('/scratch/chrischoy/Result/'); end
% save_path = ['/scratch/chrischoy/Result/' detection_result_common_name];
% [~ ,max_nms_idx] = max(ap);
% dwot_analyze_and_visualize_cnn_results(fullfile(SAVE_PATH, detection_result_file), ...
%                         detectors, VOCopts, param, nms_thresholds(max_nms_idx), true, save_path);

if ~strcmp(param.proposal_tuning_mode,'none')
    for i = 1:numel(nms_thresholds)
        nms_threshold = nms_thresholds(i);
        
        ap(i) = dwot_analyze_and_visualize_3D_object_results( fullfile(SAVE_PATH, detection_tuning_result_file), ...
                            detectors, [], param, DATA_PATH, CLASS, param.color_range, nms_threshold, false, ...
                            prediction_azimuth_rotation_direction, prediction_azimuth_offset);

        ap_tuning_save_name{i} = sprintf(['AP_%s_tuning_%d_nms_%.2f.png'],...
                            detection_result_common_name, param.proposal_tuning_mode,...
                            nms_threshold);

         print('-dpng','-r150',fullfile(SAVE_PATH, ap_tuning_save_name{i}));
    end
end

% If it runs on server copy to host
if ~isempty(server_id) && ~strcmp(server_id.num,'capri7')
    for i = 1:numel(nms_thresholds)
        system(['scp ', fullfile(SAVE_PATH, ap_save_names{i}),...
            ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/',...
            SAVE_PATH]);
        if ~strcmp(param.proposal_tuning_mode,'none')
            system(['scp ', fullfile(SAVE_PATH, ap_tuning_save_name{i}),...
                ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/',...
                SAVE_PATH]);
        end
    end
    
    system(['scp ' fullfile(SAVE_PATH, detection_result_file),...
        ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/',...
        SAVE_PATH]);
    
    if ~strcmp(param.proposal_tuning_mode,'none')
        system(['scp ' fullfile(SAVE_PATH, detection_tuning_result_file),...
            ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/',...
            SAVE_PATH]);
    end
end
