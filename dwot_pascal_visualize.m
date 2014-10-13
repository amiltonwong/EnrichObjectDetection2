[~, sys_host_name] = system('hostname');
server_id = regexp(sys_host_name, '^napoli(?<num>\d+).*','names');
if isempty(server_id)
  VOC_PATH = '/home/chrischoy/dataset/PASCAL3D+_release1.1/PASCAL/VOCdevkit/';
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
addpath('Diagnosis');

% Addpath doesn't work pwd in the code uses current directory so move to
% the directory.
curDir = pwd;
eval(['cd ' VOC_PATH]);
VOCinit;
eval(['cd ' curDir]);


DATA_SET = 'PASCAL';
COMPUTING_MODE = 1;
CLASS = 'Chair';
% CLASS = 'Bicycle';
SUB_CLASS = [];
LOWER_CASE_CLASS = lower(CLASS);
TEST_TYPE = 'val';
mkdir('Result',[LOWER_CASE_CLASS '_' TEST_TYPE]);

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

visualize_detection = true;
visualize_detector = false;

sbin = 6;
n_level = 10;
n_proposals = 5;

% Load models
% models_path = {'Mesh/Bicycle/road_bike'};
% models_name = cellfun(@(x) strrep(x, '/', '_'), models_path, 'UniformOutput', false);
% [ model_names, model_paths ] = dwot_get_cad_models('Mesh', CLASS, [], {'3ds','obj'});

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

% use_idx = ismember(model_names,models_to_use);

% model_names = model_names(use_idx);
% model_paths = model_paths(use_idx);

skip_criteria = {'empty', 'truncated','difficult','occluded'};
% skip_criteria = {'empty'};
skip_name = cellfun(@(x) x(1), skip_criteria);

%%%%%%%%%%%%%%% Set Parameters %%%%%%%%%%%%
dwot_get_default_params;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.template_initialization_mode = 0; 
param.nms_threshold = 0.4;
% param.model_paths = model_paths;
param.b_calibrate = 0;
param.n_calibration_images = 100;
param.detection_threshold = 100;
param.image_scale_factor = 2;
% param.gather_


%%%%% For Debuggin purpose only
% param.detectors              = detectors;
param.detect_pyramid_padding = 10;
%%%%%%%%%%%%

%% Make templates, these are just pointers to the templates in the detectors,
% The following code copies variables to GPU or make pointers to memory
% according to the computing mode.

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
    fprintf('%d\n',imgIdx,N_IMAGE);
    imgTic = tic;
    % read annotation
    recs(imgIdx)=PASreadrecord(sprintf(VOCopts.annopath,gtids{imgIdx}));
    
    clsinds = strmatch(LOWER_CASE_CLASS,{recs(imgIdx).objects(:).class},'exact');

    if dwot_skip_criteria(recs(imgIdx).objects(clsinds), skip_criteria); continue; end

    gt(imgIdx).BB = param.image_scale_factor * cat(1, recs(imgIdx).objects(clsinds).bbox)';
    gt(imgIdx).diff = [recs(imgIdx).objects(clsinds).difficult];
    gt(imgIdx).trunc = [recs(imgIdx).objects(clsinds).truncated];
    gt(imgIdx).occ = [recs(imgIdx).objects(clsinds).occluded];
    gt(imgIdx).det = zeros(length(clsinds),1);
    
    im = imread([VOCopts.datadir, recs(imgIdx).imgname]);
    im = imresize(im,param.image_scale_factor);
    imSz = size(im);

    for i_object = 1 : numel(clsinds)
      BB = gt(imgIdx).BB(:,i_object)';
      diff = gt(imgIdx).diff(i_object);
      trunc = gt(imgIdx).trunc(i_object);
      occ = gt(imgIdx).occ(i_object);
      gt_region = im(BB(2):BB(4), BB(1):BB(3),:);
      gt_size = size(gt_region);


      % tpIdx = logical(tp{imgIdx});
      % tpIdx = bbsNMS(:, 9) > param.min_overlap;

      % Original images
      subplot(121);
      imagesc(im); axis off; axis equal;
      rectangle('position', [BB(1) BB(2) BB(3)-BB(1) BB(4)-BB(2)],'edgecolor',[1 0 0],'LineWidth',2);
      title_string = '';
      if diff
          title_string = 'difficult';
      end
      
      if trunc
          title_string = [title_string ' truncated'];
      end
      
      if occ
          title_string = [title_string ' occluded'];
      end
      
      subplot(122);
      imagesc(gt_region); axis off; axis equal;
      title(title_string);

      drawnow;
      spaceplots();

      drawnow;

       waitforbuttonpress;            
    end
end
