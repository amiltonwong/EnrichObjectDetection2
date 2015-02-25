function [ centeredHOG, scale ] = HOGTemplate( im, param)
%WHOTEMPLATEDECOMP Summary of this function goes here
%   Detailed explanation goes here
% Nrow = N1
padding             = param.image_padding;
hog_cell_threshold  = param.hog_cell_threshold;
n_cell_limit        = param.n_cell_limit;
Mu                  = param.hog_mu;
% Gamma_GPU           = param.hog_gamma_gpu;
gammaDim            = param.hog_gamma_dim;
lambda              = param.lambda;
CG_THREASHOLD       = param.cg_threshold;
CG_MAX_ITER         = param.cg_max_iter;

%%%%%%%% Get HOG template

% create white background padding
paddedIm = padarray(im2double(im), [padding, padding, 0], 1);
% paddedIm(:,1:padding,:) = 1;
% paddedIm(:,end-padding+1 : end, :) = 1;
% paddedIm(1:padding,:,:) = 1;
% paddedIm(end-padding+1 : end, :, :) = 1;

% bounding box coordinate x1, y1, x2, y2
bbox = [1 1 size(im,2) size(im,1)] + padding;

% TODO replace it
[HOGTemplate, scale] = dwot_initialize_template(paddedIm, bbox, param);

%%%%%%%% WHO conversion using matrix decomposition

HOGTemplateSz = size(HOGTemplate);
HOGDim = HOGTemplateSz(3);

nonEmptyCells = (sum(abs(HOGTemplate),3) > hog_cell_threshold);

muSwapDim = permute(Mu,[2 3 1]);
centeredHOG = bsxfun(@minus, HOGTemplate, muSwapDim);

centeredHOG = centeredHOG .* repmat(double(nonEmptyCells),[1 1 HOGDim]);
