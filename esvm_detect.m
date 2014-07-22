function [resstruct,feat_pyramid] = esvm_detect(I, models, params)
% Localize a set of models in an image.
% function [resstruct,feat_pyramid] = esvm_detect(I, models, params)
%
% If there is a small number of models (such as in per-exemplar
% mining), then fconvblas is used for detection.  If the number is
% large, then the BLOCK feature matrix method (with a single matrix
% multiplication) is used.
%
% NOTE: These local detections can be pooled with esvm_pool_exemplars_dets.m
%
% I: Input image (or already precomputed pyramid)
% models: A cell array of models to localize inside this image
%   models{:}.model.w: Learned template
%   models{:}.model.b: Learned template's offset
% params: Localization parameters (see esvm_get_default_params.m)
%
% resstruct: Sliding window output struct with 
%   resstruct.bbs{:}: Detection boxes and pyramid locations
%   resstruct.xs{:}: Detection features
% feat_pyramid: The Feature pyramid output
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if ~iscell(models)
  models = {models};
end

if ~exist('params','var')
  params = get_default_params;
end


[rs1, t1] = esvm_detectdriver(I, models, params);
rs1 = prune_nms(rs1, params);

resstruct = rs1;
feat_pyramid = t1;
return;


function [resstruct,t] = esvm_detectdriver(I, models, ...
                                             params)

if (length(models) > params.max_models_before_block_method) ...
      || (~isempty(params.nnmode))

  [resstruct,t] = esvm_detectdriverBLOCK(I, models, ...
                                         params);
  return;
end

N_TEMPLATE = length(models);
templates = cellfun(@(x)x.model.w,models,'UniformOutput',false);

%NOTE: all exemplars in this set must have the same sbin
sbin = params.sbin;

%Compute pyramid
t.size = size(I);
[t.hog, t.scales] = esvm_pyramid(I, params);
t.padder = params.detect_pyramid_padding;
for level = 1:length(t.hog)
  t.hog{level} = padarray(t.hog{level}, [t.padder t.padder 0], 0);
end

minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), t.hog);
t.hog = t.hog(minsizes >= t.padder*2);
t.scales = t.scales(minsizes >= t.padder*2);  


resstruct.padder = t.padder;
resstruct.bbs = cell(N_TEMPLATE,1);

maxers = cell(N_TEMPLATE,1);
for q = 1:N_TEMPLATE
  maxers{q} = -inf;
end

%start with smallest level first
for level = length(t.hog):-1:1
  featr = t.hog{level};
 
  rootmatch = fconvblas(featr, templates, 1, N_TEMPLATE);
  
  rmsizes = cellfun(@(x)size(x), ...
                     rootmatch,'UniformOutput',false);
  templateIdxes = find(cellfun(@(x) prod(x), rmsizes));
  
  for exid = templateIdxes
    cur_scores = rootmatch{exid};
    [aa,indexes] = sort(cur_scores(:),'descend');
    NKEEP = sum((aa>maxers{exid}) & (aa>=params.detect_keep_threshold));
    aa = aa(1:NKEEP);
    indexes = indexes(1:NKEEP);
    if NKEEP==0
      continue
    end
    
    [uus,vvs] = ind2sub(rmsizes{exid}(1:2),...
                        indexes);
    
    scale = t.scales(level);
    
    o = [uus vvs] - t.padder;

    bbs = ([o(:,2) o(:,1) o(:,2)+size(templates{exid},2) ...
               o(:,1)+size(templates{exid},1)] - 1) * ...
             sbin/scale + 1 + repmat([0 0 -1 -1],length(uus),1);

    bbs(:,5:12) = 0;
    bbs(:,5) = (1:size(bbs,1));
    bbs(:,6) = exid;
    bbs(:,8) = scale;
    bbs(:,9) = uus;
    bbs(:,10) = vvs;
    bbs(:,12) = aa;
    
    resstruct.bbs{exid} = cat(1,resstruct.bbs{exid},bbs);
    
    if (NKEEP > 0)
      newtopk = min(params.detect_max_windows_per_exemplar,size(resstruct.bbs{exid},1));
      [aa,bb] = psort(-resstruct.bbs{exid}(:,end),newtopk);
      resstruct.bbs{exid} = resstruct.bbs{exid}(bb,:);

      %TJM: changed so that we only maintain 'maxers' when topk
      %elements are filled
      if (newtopk >= params.detect_max_windows_per_exemplar)
        maxers{exid} = min(-aa);
      end
    end    
  end
end

%fprintf(1,'\n');

function [resstruct,t] = esvm_detectdriverBLOCK(I, models, ...
                                             params)

%%HERE is the chunk version of exemplar localization

N_MODEL = length(models);
ws = cellfun(@(x)x.w,models,'UniformOutput',false);

template_size = cell2mat(cellfun(@(x) x.sz', models,'UniformOutput',false));
sizes1 = template_size(1,:);
sizes2 = template_size(2,:);

S = [max(sizes1(:)) max(sizes2(:))];
fsize = params.init_params.features();
templates = zeros(S(1),S(2),fsize,N_MODEL);
templates_x = zeros(S(1),S(2),fsize,N_MODEL);
template_masks = zeros(S(1),S(2),fsize,N_MODEL);

for i = 1:N_MODEL
  t = zeros(S(1),S(2),fsize);
  t(1:models{i}.sz(1),1:models{i}.sz(2),:) = ...
      models{i}.w;

  templates(:,:,:,i) = t;
  template_masks(:,:,:,i) = repmat(double(sum(abs(t),3)>0),[1 1 fsize]);
end

%maskmat = repmat(template_masks,[1 1 1 fsize]);
%maskmat = permute(maskmat,[1 2 4 3]);
%templates_x  = templates_x .* maskmat;

sbin = params.sbin;
[t.hog, t.scales] = esvm_pyramid(I, params);
t.padder = params.detect_pyramid_padding;
resstruct.padder = t.padder;

pyr_N = cellfun(@(x)prod([size(x,1) size(x,2)]-S+1),t.hog);
sumN = sum(pyr_N);

X = zeros(S(1)*S(2)*fsize,sumN);
offsets = cell(length(t.hog), 1);
uus = cell(length(t.hog),1);
vvs = cell(length(t.hog),1);

counter = 1;
for i = 1:length(t.hog)
  s = size(t.hog{i});
  NW = s(1)*s(2);
  ppp = reshape(1:NW,s(1),s(2));
  curf = reshape(t.hog{i},[],fsize);
  b = im2col(ppp,[S(1) S(2)]);

  offsets{i} = b(1,:);
  offsets{i}(end+1,:) = i;
  
  for j = 1:size(b,2)
   X(:,counter) = reshape (curf(b(:,j),:),[],1);
   counter = counter + 1;
  end
  
  [uus{i},vvs{i}] = ind2sub(s,offsets{i}(1,:));
end

offsets = cat(2,offsets{:});

uus = cat(2,uus{:});
vvs = cat(2,vvs{:});

% m.model.w = zeros(S(1),S(2),fsize);
% m.model.b = 0;
% temp_params = params;
% temp_params.detect_save_features = 1;
% temp_params.detect_exemplar_nms_os_threshold = 1.0;
% temp_params.max_models_before_block_method = 1;
% temp_params.detect_max_windows_per_exemplar = 28000;

% [rs] = esvm_detect(I, {m}, temp_params);
% X2=cat(2,rs.xs{1}{:});
% bbs2 = rs.bbs{1};


exemplar_matrix = reshape(templates,[],size(templates,4));
r = exemplar_matrix' * X;

resstruct.bbs = cell(N_MODEL,1);
resstruct.xs = cell(N_MODEL,1);

for exid = 1:N_MODEL

  goods = find(r(exid,:) >= params.detect_keep_threshold);
  
  if isempty(goods)
    continue
  end
  
  [sorted_scores,bb] = ...
      psort(-r(exid,goods)',...
            min(params.detect_max_windows_per_exemplar, ...
                length(goods)));
  bb = goods(bb);

  sorted_scores = -sorted_scores';

  resstruct.xs{exid} = X(:,bb);
  
  levels = offsets(2,bb);
  scales = t.scales(levels);
  curuus = uus(bb);
  curvvs = vvs(bb);
  o = [curuus' curvvs'] - t.padder;

  bbs = ([o(:,2) o(:,1) o(:,2)+size(ws{exid},2) ...
           o(:,1)+size(ws{exid},1)] - 1) .* ...
             repmat(sbin./scales',1,4) + 1 + repmat([0 0 -1 ...
                    -1],length(scales),1);
  
  bbs(:,5:12) = 0;
  bbs(:,5) = (1:size(bbs,1));
  bbs(:,6) = exid;
  bbs(:,8) = scales;
  bbs(:,9) = uus(bb);
  bbs(:,10) = vvs(bb);
  bbs(:,12) = sorted_scores;
  
  if (params.detect_add_flip == 1)
    bbs = flip_box(bbs,t.size);
    bbs(:,7) = 1;
  end
  
  resstruct.bbs{exid} = bbs;
end


if params.detect_save_features == 0
  resstruct.xs = cell(N_MODEL,1);
end
%fprintf(1,'\n');

function rs = prune_nms(rs, params)
%Prune via nms to eliminate redundant detections

%If the field is missing, or it is set to 1, then we don't need to
%process anything.  If it is zero, we also don't do NMS.
rs.bbs = cellfun(@(x)esvm_nms(x,params.nms_threshold),rs.bbs,'UniformOutput',false);

if ~isempty(rs.xs)
  for i = 1:length(rs.bbs)
    if ~isempty(rs.xs{i})
      %NOTE: the fifth field must contain elements
      rs.xs{i} = rs.xs{i}(:,rs.bbs{i}(:,5) );
    end
  end
end

