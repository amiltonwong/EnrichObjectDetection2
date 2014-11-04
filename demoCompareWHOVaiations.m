addpath('HoG');
addpath('HoG/features');
addpath('Util');
addpath('DecorrelateFeature/');
addpath('../MatlabRenderer/');
addpath('../MatlabRenderer/bin');
addpath('../MatlabCUDAConv/');
addpath('3rdParty/SpacePlot');
addpath('3rdParty/MinMaxSelection/');
addpath('Diagnosis/ihog/');
addpath('Diagnosis/ihog/internal');
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
SUB_CLASS = [];
TEST_TYPE = 'val';
LOWER_CASE_CLASS = lower(CLASS);

DEVICE_ID = 0; % 0-base indexing

if COMPUTING_MODE > 0
  gdevice = gpuDevice(DEVICE_ID + 1); % Matlab use 1 base indexing
  reset(gdevice);
  cos(gpuArray(1));
end

azs = 0:15:345;
els = 0:10:40;
fovs = [25];
yaws = -0;
n_cell_limit = [250];
lambda = [0.15];
detection_threshold = 50;

visualize_detection = true;
visualize_detector = false;

sbin = 4;
n_level = 15;
n_proposals = 5;

% Get all possible sub-classes
[ model_names, model_paths ] = dwot_get_cad_models('Mesh', 'Car',[], {'3ds','dae'});

use_idx = ismember(model_names,'Honda-Accord-3');

% models_to_use = {'atx_bike',...
%               'atx_bike_rot',...
%               'bmx_bike_4',...
%               'brooklyn_machine_works_bike',...
%               'downhill_bike',...
%               'glx_bike',...
%               'road_bike',...
%               'road_bike_rot'};

% use_idx = ismember(model_names,models_to_use);
          
model_names = model_names(use_idx);
model_paths = model_paths(use_idx);
%%%%%%%%%%%%%%% Set Parameters %%%%%%%%%%%%

skip_criteria = {'none'};
dwot_get_default_params;

param.template_initialization_mode = 0; 
param.nms_threshold = 0.4;
param.model_paths = model_paths;

param.b_calibrate = 0;      % apply callibration if > 0
param.n_calibration_images = 100; 
% Calibration mode == 'gaussian', fit gaussian
%                  == 'linear' , put 0.01% of data to be above 1.
%
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
    % param.color_range = [-inf 3:0.1:10 inf];
    param.color_range = [-inf 30:5:100 inf];
    param.detection_threshold = 40;
    param.b_calibrate = 0;
end

param.image_scale_factor = 1; % scale image accordingly and detect on the scaled image

% Tuning mode == 0, no tuning
%             == 1, MCMC
%             == 2, Breadth first search
%             == 3, Quasi-Newton method (BFGS)
param.proposal_tuning_mode = 0;


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
param.calibration_mode='linear';

%% Make Renderer
if ~exist('renderer','var') || ~exist(detector_file_name,'file')  && param.proposal_tuning_mode > 0
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


VOC_PATH = '/home/chrischoy/Dataset/VOCdevkit/';
if ismac
    VOC_PATH = '~/dataset/VOCdevkit/';
end
addpath(VOC_PATH);
addpath([VOC_PATH, 'VOCcode']);

curDir = pwd;
eval(['cd ' VOC_PATH]);
VOCinit;
eval(['cd ' curDir]);
    
%% Make Detectors
azs = 0:45:180;
els =0:20:20;
fovs = [25];
yaws =0;
i = 1;
detectors = {};
for template_initialization_mode = [0 4 5 6 7]
    param.template_initialization_mode = template_initialization_mode;
    detectors{i} = dwot_make_detectors_grid(renderer, azs, els, yaws, fovs, 1:length(model_names),...
                                LOWER_CASE_CLASS, param);
    detectors_cal{i} = dwot_calibrate_detectors(detectors{i}, LOWER_CASE_CLASS, VOCopts, param,true);
    figure(1);
    print('-dpng','-r100',sprintf('CVPR14/figures/whitening/before_%d.png',template_initialization_mode));
    
    figure(2);
    print('-dpng','-r100',sprintf('CVPR14/figures/whitening/after_%d.png',template_initialization_mode));
    
    i = i + 1;
end
   
