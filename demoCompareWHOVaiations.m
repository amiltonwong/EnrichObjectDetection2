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
n_cell_limit = [200];
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
if ~exist('renderer','var') 
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
% n_cell_limits =  [100 150 200 250 300];
n_cell_limits =  [50 100 150 200 250 300 350 400];
% TODO replace it
% 0 NZ-WHO
% 1 Constant # active cell in NZ-WHO
% 2 Decorrelate all but center only the non-zero cells
% 3 NZ-WHO but normalize by # of active cells
% 4 HOG feature
% 5 Whiten all
% 6 Whiten all but zero our empty cells
% 7 center non zero, whiten all, zero out empty
% 8 Similar to 7 but find bias heuristically
% 9 Nonzero center Decomposition zero out textureless
% 10 Standard Decomposition
template_initialization_modes = [0 5 6 9];

method = [];

for template_initialization_idx = 1:numel(template_initialization_modes)
        param.template_initialization_mode = template_initialization_modes(template_initialization_idx);
        time_generation = zeros(1,numel(n_cell_limits) );
        time_calibration = zeros(1,numel(n_cell_limits) );
        for cell_limit_idx = 1:numel(n_cell_limits)
            param.n_cell_limit = n_cell_limits( cell_limit_idx );
            tic
            dwot_make_detectors_grid(renderer, azs, els, yaws, fovs, 1:length(model_names),...
                                        LOWER_CASE_CLASS, param);
            time_generation(cell_limit_idx) = toc

%             tic
%             detectors_cal{i} = dwot_calibrate_detectors(detectors{i}, LOWER_CASE_CLASS, VOCopts, param,false);
%             time_calibration(cell_limit_idx) =  toc
            %     figure(1);
            %     print('-dpng','-r100',sprintf('CVPR14/figures/whitening/before_%d.png',template_initialization_mode));
            %     
            %     figure(2);
            %     print('-dpng','-r100',sprintf('CVPR14/figures/whitening/after_%d.png',template_initialization_mode));

            i = i + 1;
        end
        method(template_initialization_idx).time_generation = time_generation;
        method(template_initialization_idx).time_calibration = time_calibration;
end

color_map = jet(numel(template_initialization_modes));

for template_initialization_idx = 1:numel(template_initialization_modes)
    
    plot(n_cell_limits , method(template_initialization_idx).time_generation / (numel(azs) * numel(els)), 'color', color_map(template_initialization_idx,:));
    hold on;
end


% for template_initialization_idx = 1:numel(template_initialization_modes)
%      (method(template_initialization_idx).time_generation + method(template_initialization_idx).time_calibration )/ (numel(azs) * numel(els))
%     plot(n_cell_limits , (method(template_initialization_idx).time_generation + method(template_initialization_idx).time_calibration )/ (numel(azs) * numel(els)), 'color', color_map(template_initialization_idx,:));
%     hold on;
% end

