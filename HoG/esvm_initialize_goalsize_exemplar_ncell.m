function curfeats = esvm_initialize_goalsize_exemplar_ncell(I, bbox, ncell)
%% Initialize the exemplar (or scene) such that the representation
% which tries to choose a region which overlaps best with the given
% bbox and contains roughly init_params.goal_ncells cells, with a
% maximum dimension of init_params.MAXDIM
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

sbin = 8;
init_params.sbin = sbin;
init_params.MAXNCELL = ncell;

%Expand the bbox to have some minimum and maximum aspect ratio
%constraints (if it it too horizontal, expand vertically, etc)
bbox = expand_bbox(bbox,I);
bbox = max(bbox,1);
bbox([1 3]) = min(size(I,2),bbox([1 3]));
bbox([2 4]) = min(size(I,1),bbox([2 4]));

bboxWidth = bbox(4) - bbox(2);
bboxHeight = bbox(3) - bbox(1);

%Create a blank image with the exemplar inside
imSize = size(I);
Ibox = zeros(size(I,1), size(I,2));    
Ibox(bbox(2):bbox(4), bbox(1):bbox(3)) = 1;

%Get the hog feature pyramid for the entire image
interval = 10;

%Hardcoded maximum number of levels in the pyramid
MAXLEVELS = 200;

%Get the levels per octave from the parameters
sc = 2 ^(1/interval);

scale = zeros(1,MAXLEVELS);
feat = {};


for i = 1:MAXLEVELS
  scaler = 1 / sc^(i-1);
    
  if ceil(bboxWidth * scaler / sbin) * ceil(bboxHeight * scaler / sbin) >= 1.2 * ncell
    continue;
  end
  
  scale(i) = scaler;
  scaled = resizeMex(I,scale(i));
  
  feat{i} = features_pedro(scaled,sbin);
  feat{i} = padarray(feat{i}, [1 1 0], 0);   %recover lost cells!!!
  
  bndX = round((size(feat{i},2)-1)*[bbox(1)-1 bbox(3)-1]/(imSize(2)-1)) + 1;
  bndY = round((size(feat{i},1)-1)*[bbox(2)-1 bbox(4)-1]/(imSize(1)-1)) + 1;
%   bndX(1) = floor(bndX(1));
%   bndX(2) = ceil(bndX(2));
%   bndY(1) = floor(bndY(1));
%   bndY(2) = ceil(bndY(2));
  
  if (bndX(2) - bndX(1) + 1) * (bndY(2) - bndY(1) + 1) <= ncell
    curfeats = feat{i}(bndY(1):bndY(2),bndX(1):bndX(2),:);
    fprintf(1,'initialized with HOG_size = [%d %d]\n',range(bndY) + 1, range(bndX) + 1);
    return;
  end
end


%Fire inside self-image to get detection location
% [model.bb, model.x] = get_target_bb(model, I, init_params);

%Normalized-HOG initialization
% model.w = reshape(model.x,size(model.w)) - mean(model.x(:));
% 
% if isfield(init_params,'wiggle_number') && ...
%       (init_params.wiggle_number > 1)
%   savemodel = model;
%   model = esvm_get_model_wiggles(I, model, init_params.wiggle_number);
% end


function [bndX, bndY, tgtLevel] = get_matching_bbox(f_real, bbox, imSize, n_max_cell)
%Given a feature pyramid, and a segmentation mask inside Ibox, find
%the best matching region per level in the feature pyramid

for a = 1:length(f_real)  
    
  bndX = round((size(f_real{a},2)-1)*[bbox(1)-1 bbox(3)-1]/(imSize(2)-1)) + 1;
  bndY = round((size(f_real{a},1)-1)*[bbox(2)-1 bbox(4)-1]/(imSize(1)-1)) + 1;
%   bndX(1) = floor(bndX(1));
%   bndX(2) = ceil(bndX(2));
%   bndY(1) = floor(bndY(1));
%   bndY(2) = ceil(bndY(2));
  
  if (bndX(2) - bndX(1) + 1) * (bndY(2) - bndY(1) + 1) <= n_max_cell
    tgtLevel = a;
    return;
  end
end
disp('didnt find a match returning the closest level');


function bbox = expand_bbox(bbox,I)
%Expand region such that is still within image and tries to satisfy
%these constraints best
%requirements: each dimension is at least 50 pixels, and max aspect
%ratio os (.25,4)
for expandloop = 1:10000
  % Get initial dimensions
  w = bbox(3)-bbox(1)+1;
  h = bbox(4)-bbox(2)+1;
  
  if h > w*4 || w < 50
    %% make wider
    bbox(3) = bbox(3) + 1;
    bbox(1) = bbox(1) - 1;
  elseif w > h*4 || h < 50
    %make taller
    bbox(4) = bbox(4) + 1;
    bbox(2) = bbox(2) - 1;
  else
    break;
  end
  
  bbox([1 3]) = cap_range(bbox([1 3]), 1, size(I,2));
  bbox([2 4]) = cap_range(bbox([2 4]), 1, size(I,1));      
end