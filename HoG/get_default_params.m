function default_params = get_default_params(sbin, nLevel, detection_threshold)
% Return the default Exemplar-SVM detection/training parameters
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved. 
%
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sliding window detection parameters 
%(using during training and testing)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin <= 2
  detection_threshold = 20;
end
if nargin <= 1
  nLevel = 10;
end
if nargin == 0
  sbin = 4;
end

%Turn on image flips for detection/training. If enabled, processing
%happes on each image as well as its left-right flipped version.
default_params.detect_add_flip = 0;


default_params.sbin = sbin;
%Levels-per-octave defines how many levels between 2x sizes in pyramid
%(denser pyramids will have more windows and thus be slower for
%detection/training)
default_params.detect_levels_per_octave = nLevel;

%By default dont save feature vectors of detections (training turns
%this on automatically)
default_params.detect_save_features = 0;

%Default detection threshold (negative margin makes most sense for
%SVM-trained detectors).  Only keep detections for detection/training
%that fall above this threshold.
default_params.detect_keep_threshold = -1;

%Maximum #windows per exemplar (per image) to keep
% default_params.detect_max_windows_per_exemplar = 10;

%Determines if NMS (Non-maximum suppression) should be used to
%prune highly overlapping, redundant, detections.
%If less than 1.0, then we apply nms to detections so that we don't have
%too many redundant windows [defaults to 0.5]
%NOTE: mining is much faster if this is turned off!
default_params.nms_threshold = 0.5;

%How much we pad the pyramid (to let detections fall outside the image)
default_params.detect_pyramid_padding = 5;

%The maximum scale to consdider in the feature pyramid
default_params.detect_max_scale = 1.0;

%The minimum scale to consider in the feature pyramid
default_params.detect_min_scale = .01;

% Number of NMSed detections per exemplar
default_params.detect_max_windows_per_exemplar = 40;

%Only keep detections that have sufficient overlap with the input's
%global bounding box.  If greater than 0, then only keep detections
%that have this OS with the entire input image.
default_params.detect_min_scene_os = 0.0;

% Choose the number of images to process in each chunk for detection.
% This parameters tells us how many images each core will process at
% at time before saving results.  A higher number of images per chunk
% means there will be less constant access to hard disk by separate
% processes than if images per chunk was 1.
default_params.detect_images_per_chunk = 4;

%NOTE: If the number of specified models is greater than 20, use the
%BLOCK-based method
default_params.max_models_before_block_method = 20;

default_params.detection_threshold = detection_threshold;

%Initialize framing function
init_params.features = @esvm_features;
init_params.sbin = sbin;
init_params.goal_ncells = 100;

default_params.init_params = init_params;