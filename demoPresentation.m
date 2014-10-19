% Demo
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
% CLASS = 'Bicycle';
SUB_CLASS = [];
LOWER_CASE_CLASS = lower(CLASS);
TEST_TYPE = 'val';
mkdir('Result',[LOWER_CASE_CLASS '_' TEST_TYPE]);

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

sbin = 6;
n_level = 15;

% Load models
[ model_names, model_paths ] = dwot_get_cad_models('Mesh', CLASS, [], {'3ds','obj'});

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

%%%%%%%%%%%%%%% Set Parameters %%%%%%%%%%%%
dwot_get_default_params;
param.color_range = [-inf 100:20:300 inf];




%% Initialize renderer
renderer = Renderer();
if ~renderer.initialize(model_paths, 700, 700, 0, 0, 0, 0, 25)
  error('fail to load model');
end

%% Render a model
az = 45; el = 7; yaw = 0; fov = 25; model_index = 7;
renderer.setModelIndex(model_index);
renderer.setViewpoint(az, el, yaw, 0, fov);
[rendering, depth ] = renderer.renderCrop();

figure(1);
colormap gray;
subplot(121); imagesc(rendering); axis equal; axis off;
subplot(122); imagesc(1-depth); axis equal; axis off;
spaceplots();

%% Feature extraction
[ WHO, HOG] = WHOTemplateCG_CUDA(rendering, param);
% [ WHO, HOG] = WHOTemplateCG_GPU(rendering, param);
figure(2);
colormap gray;
subplot(311); imagesc(rendering);  axis equal; axis off;
subplot(312); imagesc(-HOGpicture(HOG)); axis equal; axis off;
subplot(313); imagesc(-HOGpicture(WHO)); axis equal; axis off;
spaceplots();

detector = dwot_get_detector(renderer, az, el, yaw, fov, ...
                    model_index, 'not_supported_model_class', param);
                  
template_gpu = {gpuArray(detector.whow(end:-1:1,end:-1:1,:))};
[bbsAllLevel, hog, scales] = dwot_detect_gpu( im, template_gpu, param);