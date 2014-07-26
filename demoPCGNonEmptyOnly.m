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
n_cell_limit = 150;
lambda = 0.01;
CG_THREASHOLD = 10^-3;
N_CIRC_BLOCKS = 6;
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

% TODO replace it
HOGTemplate = esvm_initialize_goalsize_exemplar_ncell(paddedIm, bbox, n_cell_limit);

%%%%%%%% WHO conversion using matrix decomposition

HOGTemplatesz = size(HOGTemplate);
wHeight = int16(HOGTemplatesz(1));
wWidth = int16(HOGTemplatesz(2));
HOGDim = int16(HOGTemplatesz(3));
nonEmptyCells = (sum(HOGTemplate,3) > hog_cell_threshold);
idxNonEmptyCells = int16(find(nonEmptyCells));
[nonEmptyRows,nonEmptyCols] = ind2sub([wHeight, wWidth], idxNonEmptyCells);
nonEmptyRows = int16(nonEmptyRows);
nonEmptyCols = int16(nonEmptyCols);

n_non_empty_cells = int16(numel(nonEmptyRows));

sigmaDim = n_non_empty_cells * HOGDim;

CirculantBlocks = cell(1, N_CIRC_BLOCKS);
CirculantBlocks = cellfun(@(x) zeros(HOGDim, HOGDim, 'single'), CirculantBlocks, 'UniformOutput', false);

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
      
    circIdx = mod(gammaRowIdx - gammaColIdx,n_non_empty_cells) + 1;
    if circIdx <= N_CIRC_BLOCKS
      CirculantBlocks{circIdx} = CirculantBlocks{circIdx} + ...
          Gamma((gammaRowIdx-1)*HOGDim + 1 : gammaRowIdx*HOGDim , (gammaColIdx - 1)*HOGDim + 1 : gammaColIdx*HOGDim);
    end
  end
end

% 
% rangeHOGDim = 1:HOGDim;
% for cellIdx = 0:HOGDim:((n_non_empty_cells-1)*HOGDim)
%   rowIdx = nonEmptyRows(cellIdx/HOGDim + 1); % sub2ind([wHeight, wWidth],i,j);
%   colIdx = nonEmptyCols(cellIdx/HOGDim + 1);
%   
%   for otherCellIdx = 0:HOGDim:((n_non_empty_cells-1)*HOGDim)
%     gammaRowIdx = abs(rowIdx - nonEmptyRows(otherCellIdx/HOGDim + 1)) + 1;
%     gammaColIdx = abs(colIdx - nonEmptyCols(otherCellIdx/HOGDim + 1)) + 1;
%     
%     Sigma(cellIdx + rangeHOGDim, otherCellIdx + rangeHOGDim) = ...
%         Gamma((gammaRowIdx-1)*HOGDim + rangeHOGDim, (gammaColIdx - 1)*HOGDim + rangeHOGDim);
%       
%     circIdx = mod(gammaRowIdx - gammaColIdx,n_non_empty_cells) + 1;
%     if circIdx <= N_CIRC_BLOCKS
%       CirculantBlocks{circIdx} = CirculantBlocks{circIdx} + ...
%           Gamma((gammaRowIdx-1)*HOGDim + rangeHOGDim, (gammaColIdx - 1)*HOGDim + rangeHOGDim);
%     end
%   end
% end

CirculantBlocks = cellfun(@(x) x/single(n_non_empty_cells), CirculantBlocks, 'UniformOutput', false);
% imagesc(cell2mat(CirculantBlocks)); axis equal; colorbar;
%%%%%%%%%%%%%%%% Preconditioner
% P = sparse(1:double(sigmaDim), mod((1:double(sigmaDim)) + double(HOGDim) - 1, double(sigmaDim)) + 1, ones(1,sigmaDim));
% BlockI = sparse(double(sigmaDim), double(sigmaDim));
% for cellIdx = 1:n_non_empty_cells
%   blockRange = (cellIdx - 1) * HOGDim + (1:HOGDim);
%   BlockI(blockRange, blockRange) = 1;
% end
% 
% CirculantBlocks = cell(1, ceil(n_non_empty_cells / 2) );
% for cellIdx = 1:ceil(n_non_empty_cells/2)
%   CirculantBlocks{cellIdx} = BlockI .* Sigma
% 
% %%%%%%%%%%%%%%%% Vectorized Sigma
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

toc

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
toc


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
toc

figure(1); imagesc(HOGpicture(abs(WHOTemplate_CG))); colorbar;
figure(2); imagesc(HOGpicture(abs(WHOTemplate))); colorbar;
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