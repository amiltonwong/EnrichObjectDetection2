addpath('HoG');
addpath('HoG/features');
addpath('Util');
addpath('../MatlabRenderer/');
addpath('DecorrelateFeature/');

COMPUTING_MODE = 1;
CLASS = 'bicycle';
TYPE  = 'val';
if COMPUTING_MODE > 0
  gdevice = gpuDevice(1);
  reset(gdevice);
  cos(gpuArray(1));
end

n_cell_limit = [190];
lambda = [0.015];

visualize_detection = true;
visualize_detector = false;
% visualize = false;

sbin = 4;
n_level = 10;
detection_threshold = 120;
n_proposals = 1;

models_path = {'Mesh/Bicycle/road_bike'};
models_name = cellfun(@(x) strrep(x, '/', '_'), models_path, 'UniformOutput', false);

dwot_get_default_params;
param.models_path = models_path;

renderer = Renderer();
if ~renderer.initialize([models_path{1} '.3ds'], 700, 700, 0, 0, 0, 0, 25)
  error('fail to load model');
end

try
  renderer.setViewpoint(90,0,0,0,25);
  im = renderer.renderCrop();

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
  paddedIm = padarray(im2double(im), [padding, padding, 0]);
  paddedIm(:,1:padding,:) = 1;
  paddedIm(:,end-padding+1 : end, :) = 1;
  paddedIm(1:padding,:,:) = 1;
  paddedIm(end-padding+1 : end, :, :) = 1;

  % bounding box coordinate x1, y1, x2, y2
  bbox = [1 1 size(im,2) size(im,1)] + padding;
  HOGTemplate = dwot_initialize_template_const_active_cell(paddedIm, bbox, n_cell_limit, hog_cell_threshold);

  
  %%%%%%%% WHO conversion using matrix decomposition
  HOGTemplateSz = size(HOGTemplate);
  wHeight = HOGTemplateSz(1);
  wWidth = HOGTemplateSz(2);
  HOGDim = HOGTemplateSz(3);
  nonEmptyCells = (sum(HOGTemplate,3) > hog_cell_threshold);
  idxNonEmptyCells = find(nonEmptyCells);
  [nonEmptyRows,nonEmptyCols] = ind2sub([wHeight, wWidth], idxNonEmptyCells);
  nonEmptyRows = int32(nonEmptyRows);
  nonEmptyCols = int32(nonEmptyCols);

  n_non_empty_cells = int32(numel(nonEmptyRows));

  sigmaDim = n_non_empty_cells * HOGDim;
  SigmaGPU = zeros(sigmaDim, sigmaDim, 'single', 'gpuArray');
  param.scramble_kernel.GridSize = [ceil(double(sigmaDim)/param.N_THREAD_H ), ceil(double(sigmaDim)/param.N_THREAD_W ), 1];

  nonEmptyRowsGPU = gpuArray(nonEmptyRows - 1);
  nonEmptyColsGPU = gpuArray(nonEmptyCols - 1);

  AGPU = feval(param.scramble_kernel, SigmaGPU, param.hog_gamma_gpu, single(lambda), nonEmptyRowsGPU, nonEmptyColsGPU, gammaDim(1), HOGDim, n_non_empty_cells);
  fprintf('start decorrelation\n');
  A_kernel = gather(AGPU);
  A_mex = cudaDecorrelateFeature(param.hog_gamma_gpu, single(HOGTemplate), nonEmptyRows, nonEmptyCols, lambda);
catch e
  disp(e.message);
end
renderer.delete();