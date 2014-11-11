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
rng('default');
DATA_SET = 'PASCAL12';
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
fovs = [25];
yaws = 0;
n_cell_limit = [250];
lambda = [0.15];
detection_threshold = 80;

visualize_detection = true;
visualize_detector = false;

sbin = 6;
n_level = 20;
n_max_proposals = 10;
n_max_tuning = 1;
% Load models
% models_path = {'Mesh/Bicycle/road_bike'};
% models_name = cellfun(@(x) strrep(x, '/', '_'), models_path, 'UniformOutput', false);
[ model_names, model_paths ] = dwot_get_cad_models('Mesh', CLASS, [], {'3ds','obj'});

% models_to_use = {'bmx_bike',...
%               'fixed_gear_road_bike',...
%               'glx_bike',...
%               'road_bike'};

% models_to_use = {'2012-VW-beetle-turbo',...
%               'Kia_Spectra5_2006',...
%               '2008-Jeep-Cherokee',...
%               'Ford Ranger Updated',...
%               'BMW_X1_2013',...
%               'Honda_Accord_Coupe_2009',...
%               'Porsche_911',...
%               '2009 Toyota Cargo'};

models_to_use = {'Honda_Accord_Coupe_2009'};

use_idx = ismember(model_names,models_to_use);

model_names = model_names(use_idx);
model_paths = model_paths(use_idx);

% skip_criteria = {'empty', 'truncated','difficult'};
skip_criteria = {'none'};
skip_name = cellfun(@(x) x(1), skip_criteria);

%%%%%%%%%%%%%%% Set Parameters %%%%%%%%%%%%
dwot_get_default_params;

param.template_initialization_mode = 0; 
param.nms_threshold = 0.3;
param.model_paths = model_paths;

param.b_calibrate = 0;      % apply callibration if > 0
param.n_calibration_images = 100; 
param.calibration_mode = 'gaussian';

param.detection_threshold = 30;
param.image_scale_factor = 2; % scale image accordingly and detect on the scaled image

% Tuning mode == 'none', no tuning
%             == 'mcm', MCMC
%             == 'not supported yet', Breadth first search
%             == 'bfgs', Quasi-Newton method (BFGS)
param.proposal_tuning_mode = 'none';

% Detection mode == 'dwot' ours
%                == 'cnn'
%                == 'dpm'
param.detection_mode = 'cnn';

% Save mode = 'd' detections
%           = 'dt' detection and template index
%           = 'dv' detection and viewpoint
%           = 'dtp' detection template index and proposal
%           = 'dtpv' detection template index and proposal
%           = 'dvpv' detection template index and proposal
%           = 'dvp' detection viewpoint and proposal
param.detection_save_mode = 'dv';
param.proposal_dwot_save_mode = 'dtpv';
param.tuning_save_mode = 'dvpv';
param.export_figure = false;

% image_region_extraction.padding_ratio = 0.2;
param.PASCAL3D_ANNOTATION_PATH = PASCAL3D_ANNOTATION_PATH;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param.color_range = [-inf 20:5:100 inf];
param.cnn_color_range = [ -inf -4:0.1:3 inf];


% AP unit test
vnum = 8;
[recall, precision, accuracy, ap, aa] = compute_recall_precision_accuracy_orig('car','/home/chrischoy/Dataset/3DDPM_result/car_vp08.mat',vnum,VOCopts, param);

detection_result_txt = 'Result/car_val/PASCAL12_car_val_init_0_Car_each_27_lim_250_lam_0.150_a_24_e_3_y_1_f_1_scale_2.00_sbin_6_level_15_skp_n_nms_0.40_server_103_sm_dtpv_vocdpm_proposal_dwot_tmp_10.txt';

detection_result_txt = 'Result/car_val/PASCAL12_car_val_init_0_Car_each_27_lim_250_lam_0.150_a_24_e_3_y_1_f_1_scale_2.00_sbin_6_level_15_skp_n_nms_0.40_sm_dtpv_server_103_vocdpm_proposal_dwot_tmp_7.txt';
[recall, precision, accuracy, ap, aa] = compute_recall_precision_accuracy(detection_result_txt, vnum, VOCopts, param);

dwot_analyze_and_visualize_cnn_results(detection_result_txt, detectors,VOCopts, param, 0.7, false, [], 8, -1, 0)

detection_result_cnn_txt = 'Result/car_val/PASCAL12_car_val_init_0_Car_each_27_lim_250_lam_0.150_a_24_e_3_y_1_f_1_scale_2.00_sbin_6_level_15_skp_n_nms_0.40_server_103_sm_dtp_cnn_proposal_dwot_tmp_9.txt';

dwot_analyze_and_visualize_cnn_results(detection_result_cnn_txt, detectors,VOCopts, param, 0.7, false, [], 8, -1, 0)


chair_result_txt = 'Result/chair_val/PASCAL12_chair_val_init_0_Chair_each_200_lim_250_lam_0.150_a_31_e_2_y_1_f_1_scale_2.00_sbin_6_level_15_skp_etdo_server_104_sm_dt_model_400_and_800_tmp_3.txt';
chair_result_txt = 'Result/chair_val/PASCAL12_chair_val_init_0_Chair_each_200_lim_250_lam_0.150_a_31_e_2_y_1_f_1_scale_2.00_sbin_6_level_15_skp_etdo_server_104_sm_dt_model_800_tmp_3.txt';
dwot_analyze_and_visualize_cnn_results(chair_result_txt, detectors,VOCopts, param, 0.7, false, [], 8, -1, 0)
dwot_analyze_and_visualize_pascal_results(chair_result_txt, detectors, [], VOCopts, param, {'empty','difficult','occluded','truncated'}, param.color_range, 0.5,false)
