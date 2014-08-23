%Turn on image flips for detection/training. If enabled, processing
%happes on each image as well as its left-right flipped version.
param.detect_add_flip = 0;


param.sbin = sbin;
%Levels-per-octave defines how many levels between 2x sizes in pyramid
%(denser pyramids will have more windows and thus be slower for
%detection/training)
param.detect_levels_per_octave = n_level;

%By default dont save feature vectors of detections (training turns
%this on automatically)
% default_params.detect_save_features = 0;

%Default detection threshold (negative margin makes most sense for
%SVM-trained detectors).  Only keep detections for detection/training
%that fall above this threshold.
% default_params.detect_keep_threshold = -1;

%Maximum #windows per exemplar (per image) to keep
% default_params.detect_max_windows_per_exemplar = 10;

%Determines if NMS (Non-maximum suppression) should be used to
%prune highly overlapping, redundant, detections.
%If less than 1.0, then we apply nms to detections so that we don't have
%too many redundant windows [defaults to 0.5]
%NOTE: mining is much faster if this is turned off!
param.nms_threshold = 0.5;

param.min_overlap = 0.5;

%How much we pad the pyramid (to let detections fall outside the image)
param.detect_pyramid_padding = 5;

% minimum image hog length that we use for convolution
param.min_hog_length = 10;

%The maximum scale to consdider in the feature pyramid
param.detect_max_scale = 1.0;

%The minimum scale to consider in the feature pyramid
param.detect_min_scale = .01;

% Number of NMSed detections per exemplar
% default_params.detect_max_windows_per_exemplar = 40;

%Only keep detections that have sufficient overlap with the input's
%global bounding box.  If greater than 0, then only keep detections
%that have this OS with the entire input image.
% default_params.detect_min_scene_os = 0.0;

% Choose the number of images to process in each chunk for detection.
% This parameters tells us how many images each core will process at
% at time before saving results.  A higher number of images per chunk
% means there will be less constant access to hard disk by separate
% processes than if images per chunk was 1.
% default_params.detect_images_per_chunk = 4;

%NOTE: If the number of specified models is greater than 20, use the
%BLOCK-based method
% default_params.max_models_before_block_method = 20;

param.detection_threshold = detection_threshold;

%Initialize framing function
init_params.features = @esvm_features;
init_params.sbin = sbin;
% init_params.goal_ncells = 100;

param.init_params = init_params;

%% WHO setting
% TEMPLATE_INITIALIZATION_MODE == 0
%     Creates templates that have approximately same number of cells
% TEMPLATE_INITIALIZATION_MODE == 1
%     Creates templates that have approxmiately same number of active cells
% Active cells are the HOG cells whose absolute values is above the
% HOG_CELL_THRESHOLD
param.template_initialization_mode = 0; 
param.image_padding       = 50;
param.lambda              = lambda;
param.n_level_per_octave  = n_level;
param.detection_threshold = detection_threshold;
param.n_cell_limit        = n_cell_limit;
param.class               = CLASS;
param.type                = TYPE;
param.hog_cell_threshold  = 1.0;
param.feature_dim         = 31;

% Statistics
stats = load('Statistics/sumGamma_N1_40_N2_40_sbin_4_nLevel_10_nImg_1601_napoli1_gamma.mat');

param.hog_mu          = stats.mu;
param.hog_gamma       = stats.Gamma;
param.hog_gamma_gpu   = gpuArray(single(stats.Gamma));
param.hog_gamma_dim   = size(stats.Gamma);

%% CG setting
param.N_THREAD_H = 32;
param.N_THREAD_W = 32;

param.scramble_gamma_to_sigma_file = './scrambleGammaToSigma';
if ~exist([param.scramble_gamma_to_sigma_file  '.ptx'],'file')
    system(['nvcc -ptx ' param.scramble_gamma_to_sigma_file '.cu']);
end
scramble_kernel                  = parallel.gpu.CUDAKernel([param.scramble_gamma_to_sigma_file '.ptx'],[param.scramble_gamma_to_sigma_file '.cu']);
scramble_kernel.ThreadBlockSize  = [param.N_THREAD_H , param.N_THREAD_W , 1];
param.scramble_kernel = scramble_kernel;
  
param.cg_threshold        = 10^-3;
param.cg_max_iter         = 60;

param.computing_mode = COMPUTING_MODE;

%% Region Extraction
param.region_extraction_padding_ratio = 0.2;
param.region_extraction_levels = 2;
% MCMC Setting
param.mcmc_max_iter = 50;

%% Cuda Convolution Params
% THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D, THREAD_PER_BLOCK_2D
param.cuda_conv_n_threads = [8, 8, 4, 8];

