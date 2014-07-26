% Load image
addpath('HoG');
addpath('HoG/features');
addpath('Util');

% Gives im, depth
demoRendering;

if ~exist('Gamma','var')
  load('Statistics/sumGamma_N1_40_N2_40_sbin_4_nLevel_10_nImg_1601_napoli1_gamma.mat');
end

hog_cell_threshold = 1.5 * 10^0;
padding = 20;
n_cell_limits = [50 100 150 200 250 300 350 400];
lambda = 0.01;

preprocess_time_per_case = zeros(1,numel(n_cell_limits));
decomp_time_per_case = zeros(1,numel(n_cell_limits));
cg_time_per_case = zeros(1,numel(n_cell_limits));
decomp_residual_per_case = zeros(1,numel(n_cell_limits));
cg_residual_per_case = zeros(1,numel(n_cell_limits));

CG_THREASHOLD = 10^-4;
%%%%%%%% Get HOG template

tic
% create white background padding
paddedIm = padarray(im2double(im), [padding, padding, 0]);
paddedIm(:,1:padding,:) = 1;
paddedIm(:,end-padding+1 : end, :) = 1;
paddedIm(1:padding,:,:) = 1;
paddedIm(end-padding+1 : end, :, :) = 1;

% bounding box coordinate x1, y1, x2, y2
bbox = [1 1 size(im,2) size(im,1)] + padding;


for caseIdx = 1:numel(n_cell_limits)
  n_cell_limit = n_cell_limits(caseIdx);
  
  tic;
  HOGTemplate = esvm_initialize_goalsize_exemplar_ncell(paddedIm, bbox, n_cell_limit);

  %%%%%%%% WHO conversion using matrix decomposition

  HOGTemplatesz = size(HOGTemplate);
  wHeight = HOGTemplatesz(1);
  wWidth = HOGTemplatesz(2);
  HOGDim = HOGTemplatesz(3);
  nonEmptyCells = (sum(HOGTemplate,3) > hog_cell_threshold);
  idxNonEmptyCells = find(nonEmptyCells);
  [nonEmptyRows,nonEmptyCols] = ind2sub([wHeight, wWidth], idxNonEmptyCells);
  nonEmptyRows = int16(nonEmptyRows);
  nonEmptyCols = int16(nonEmptyCols);

  n_non_empty_cells = int16(numel(nonEmptyRows));

  sigmaDim = n_non_empty_cells * HOGDim;

  % CircSigma = zeros(sigmaDim, sigmaDim, 'single');
  Sigma = zeros(sigmaDim, sigmaDim, 'single');

  for cellIdx = 1:n_non_empty_cells
    rowIdx = nonEmptyRows(cellIdx); % sub2ind([wHeight, wWidth],i,j);
    colIdx = nonEmptyCols(cellIdx);
    for otherCellIdx = 1:n_non_empty_cells
  %     otherRowIdx = nonEmptyRows(otherCellIdx);
  %     otherColIdx = nonEmptyCols(otherCellIdx);
  %     gammaRowIdx = abs(rowIdx - otherRowIdx) + 1;
  %     gammaColIdx = abs(colIdx - otherColIdx) + 1;

      gammaRowIdx = abs(rowIdx - nonEmptyRows(otherCellIdx)) + 1;
      gammaColIdx = abs(colIdx - nonEmptyCols(otherCellIdx)) + 1;

      Sigma((cellIdx-1)*HOGDim + 1:cellIdx * HOGDim, (otherCellIdx-1)*HOGDim + 1:otherCellIdx*HOGDim) = ...
          Gamma((gammaRowIdx-1)*HOGDim + 1 : gammaRowIdx*HOGDim , (gammaColIdx - 1)*HOGDim + 1 : gammaColIdx*HOGDim);
    end
  end

  %%%%%%%%%%%%%%%% Vectorized Sigma
  % gammaRelRow
  % gammaRelCol 
  % for cellIdx = 1:n_non_empty_cells
  %   rowIdx = nonEmptyRows(cellIdx); % sub2ind([wHeight, wWidth],i,j);
  %   colIdx = nonEmptyCols(cellIdx);
  %   for otherCellIdx = 1:n_non_empty_cells
  % %     otherRowIdx = nonEmptyRows(otherCellIdx);
  % %     otherColIdx = nonEmptyCols(otherCellIdx);
  % %     gammaRowIdx = abs(rowIdx - otherRowIdx) + 1;
  % %     gammaColIdx = abs(colIdx - otherColIdx) + 1;
  %     
  %     gammaRowIdx = abs(rowIdx - nonEmptyRows(otherCellIdx)) + 1;
  %     gammaColIdx = abs(colIdx - nonEmptyCols(otherCellIdx)) + 1;
  %     
  %     Sigma((cellIdx-1)*HOGDim + 1:cellIdx * HOGDim, (otherCellIdx-1)*HOGDim + 1:otherCellIdx*HOGDim) = ...
  %         Gamma((gammaRowIdx-1)*HOGDim + 1 : gammaRowIdx*HOGDim , (gammaColIdx - 1)*HOGDim + 1 : gammaColIdx*HOGDim);
  %   end
  % end

  preprocess_time_per_case(caseIdx) = toc

  %%%%%%%%%%%%% Conjugate Gradient 
  muSwapDim = permute(single(mu),[2 3 1]);
  centeredHOG = bsxfun(@minus, HOGTemplate, muSwapDim);
  permHOG = permute(centeredHOG,[3 1 2]); % [HOGDim, Nrow, Ncol] = HOGDim, N1, N2
  onlyNonEmptyIdx = cell2mat(arrayfun(@(x) x + (1:HOGDim)', HOGDim * (idxNonEmptyCells - 1),'UniformOutput',false));
  nonEmptyHOG = permHOG(onlyNonEmptyIdx);

  tic
  x = zeros(sigmaDim,1,'single');
  % x = 100 * nonEmptyHOG;
  b = nonEmptyHOG;
  A = Sigma + single(lambda) * eye(sigmaDim,'single');
  r = b;
  r_start_norm = r' * r;
  % r = b - A * x;
  d = r;

  n_cache = 1;
  x_cache = zeros(sigmaDim,n_cache,'single');
  r_norm_cache = ones(1, n_cache) * inf;

  MAX_ITER = 6 * 10^1;
  r_hist = zeros(1, MAX_ITER,'single');
  i = 0;
  r_norm_next = inf;
  while i < MAX_ITER
    i = i + 1;

    r_norm = (r'*r);
    r_hist(i) = r_norm/r_start_norm;

    if r_norm/r_start_norm < CG_THREASHOLD
      break;
    end

    Ad = A * d;
    alpha = r_norm/(d' * Ad);
    x = x + alpha * d;
    r = r - alpha * Ad;
    beta = r'*r/r_norm;
    d = r + beta * d;
  end

  if i == MAX_ITER
    disp('fail to get x within threshold');
  end

  WHOTemplate_CG = zeros(prod(HOGTemplatesz),1);
  WHOTemplate_CG(onlyNonEmptyIdx) = x(:,1);
  WHOTemplate_CG =  reshape(WHOTemplate_CG,[HOGDim, wHeight, wWidth]);
  WHOTemplate_CG = permute(WHOTemplate_CG,[2,3,1]);
  cg_time_per_case(caseIdx) = toc


  %%%%%%%%%%%%% Compare with Decomposition Method.
  tic
  success = false;
  firstTry = true;
  while ~success
    if firstTry
        Sigma = Sigma + lambda * eye(size(Sigma));
    else
        Sigma = Sigma + 0.01 * eye(size(Sigma));
    end
    firstTry = false;
    [R, p] = chol(Sigma);
    if p == 0
      success = true;
    else
      display('trying decomposition again');
    end
  end


  sigInvCenteredWs = R\(R'\nonEmptyHOG);
  WHOTemplate = zeros(prod(HOGTemplatesz),1);
  WHOTemplate(onlyNonEmptyIdx) = sigInvCenteredWs;
  WHOTemplate =  reshape(WHOTemplate,[HOGDim, wHeight, wWidth]);
  WHOTemplate = permute(WHOTemplate,[2,3,1]);
  decomp_time_per_case(caseIdx) = toc
  
  cg_residual_per_case(caseIdx) = norm(b - A * x)
  decomp_residual_per_case(caseIdx) = norm(b - A * sigInvCenteredWs)
end

figure(1); imagesc(HOGpicture(abs(WHOTemplate_CG))); colorbar; title('Conjugate Gradient nonzero cells');
figure(2); imagesc(HOGpicture(abs(WHOTemplate))); colorbar; title('Decomposition nonzero cells');

figure(3); plot(n_cell_limits, cg_time_per_case,'-+'); title('Conjugate Gradient/Decomposition nonzero cells'); xlabel('N Cells'); ylabel('seconds');
hold on;   plot(n_cell_limits, decomp_time_per_case,'r-o'); 
hold on;   plot(n_cell_limits, preprocess_time_per_case,'k-x');
legend('CG','Decomp','Preprocessing');

figure(4); plot(n_cell_limits, cg_residual_per_case, '-+'); title('Conjugate Gradient/Decomposition nonzero cells'); xlabel('N Cells'); ylabel('residuals');
hold on; plot(n_cell_limits, decomp_residual_per_case, 'r-o'); legend('CG','Decomp');

% imagesc(HOGpicture(abs(WHOTemplate))); colorbar; title('Decomposition all cells');

% figure; imagesc(HOGpicture(abs(WHOTemplate - WHOTemplate_CG))); colorbar;

% err = sum(abs(WHOTemplate(:) - WHOTemplate_CG(:)))
% scale = sum(abs(WHOTemplate(:)))
% 
% err/scale

%%%%%%%%%% Numerical Stability
% norm(b - A * x)
% norm(b - A * sigInvCenteredWs)
% 
% figure; plot(log10(r_hist))