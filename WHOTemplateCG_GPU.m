function [ WHOTemplate_CG, HOGTemplate, r_hist, residual] = WHOTemplateCG_GPU( im, scrambleKernel, Mu, Gamma_GPU, n_cell_limit, lambda, padding, hog_cell_threshold, CG_THREASHOLD, CG_MAX_ITER, N_THREAD)
%WHOTEMPLATEDECOMP Summary of this function goes here
%   Detailed explanation goes here
% Nrow = N1

if nargin < 10
  N_THREAD = 32;
end

if nargin < 9
  CG_MAX_ITER = 6 * 10^1;
end

if nargin < 8
  CG_THREASHOLD = 10^-3;
end

if nargin < 7
  hog_cell_threshold = 1.5 * 10^0;
end

if nargin < 6
  padding = 50;
end


%%%%%%%% Get HOG template

% create white background padding
paddedIm = padarray(im2double(im), [padding, padding, 0]);
paddedIm(:,1:padding,:) = 1;
paddedIm(:,end-padding+1 : end, :) = 1;
paddedIm(1:padding,:,:) = 1;
paddedIm(end-padding+1 : end, :, :) = 1;

% bounding box coordinate x1, y1, x2, y2
bbox = [1 1 size(im,2) size(im,1)] + padding;

% TODO replace it
HOGTemplate = dwot_initialize_template(paddedIm, bbox, n_cell_limit);

%%%%%%%% WHO conversion using matrix decomposition

HOGTemplatesz = size(HOGTemplate);
wHeight = HOGTemplatesz(1);
wWidth = HOGTemplatesz(2);
HOGDim = HOGTemplatesz(3);
nonEmptyCells = (sum(HOGTemplate,3) > hog_cell_threshold);
idxNonEmptyCells = find(nonEmptyCells);
[nonEmptyRows,nonEmptyCols] = ind2sub([wHeight, wWidth], idxNonEmptyCells);
nonEmptyRows = int32(nonEmptyRows);
nonEmptyCols = int32(nonEmptyCols);

n_non_empty_cells = int32(numel(nonEmptyRows));

sigmaDim = n_non_empty_cells * HOGDim;
scrambleKernel.GridSize = [ceil(double(sigmaDim)/N_THREAD ), ceil(double(sigmaDim)/N_THREAD ), 1];

SigmaGPU = zeros(sigmaDim, sigmaDim, 'single', 'gpuArray');

nonEmptyRowsGPU = gpuArray(nonEmptyRows - 1);
nonEmptyColsGPU = gpuArray(nonEmptyCols - 1);

HOGDimGPU = gpuArray(int32(HOGDim));
n_non_empty_cellsGPU = gpuArray(n_non_empty_cells);

AGPU = feval(scrambleKernel, SigmaGPU, Gamma_GPU, single(lambda), nonEmptyRowsGPU, nonEmptyColsGPU, gammaDimGPU, HOGDimGPU, n_non_empty_cellsGPU);
  
muSwapDim = permute(Mu,[2 3 1]);
centeredHOG = bsxfun(@minus, HOGTemplate, muSwapDim);
permHOG = permute(centeredHOG,[3 1 2]); % [HOGDim, Nrow, Ncol] = HOGDim, N1, N2
onlyNonEmptyIdx = cell2mat(arrayfun(@(x) x + (1:HOGDim)', HOGDim * (idxNonEmptyCells - 1),'UniformOutput',false));
nonEmptyHOG = permHOG(onlyNonEmptyIdx);
nonEmptyHOGGPU = gpuArray(single(nonEmptyHOG));

x = zeros(sigmaDim,1,'single','gpuArray');
b = nonEmptyHOGGPU;
r = b;
r_start_norm = r' * r;
d = b;

r_hist = zeros(1, CG_MAX_ITER,'single','gpuArray');
i = 0;
while i < CG_MAX_ITER
  i = i + 1;

  r_norm = (r'*r);
  r_hist(i) = r_norm/r_start_norm;

  if r_hist(i) < CG_THREASHOLD
    break;
  end

  Ad = AGPU * d;
  alpha = r_norm/(d' * Ad);
  x = x + alpha * d;
  r = r - alpha * Ad;
  beta = (r'*r)/r_norm;
  d = r + beta * d;
end

if i == MAX_ITER
  disp('fail to get x within threshold');
end


WHOTemplate_CG = zeros(prod(HOGTemplateSz),1);
WHOTemplate_CG(onlyNonEmptyIdx) = gather(x);
WHOTemplate_CG =  reshape(WHOTemplate_CG,[HOGDim, wHeight, wWidth]);
WHOTemplate_CG = permute(WHOTemplate_CG,[2,3,1]);

if nargout > 3
  residual = norm(b-AGPU*x);
end