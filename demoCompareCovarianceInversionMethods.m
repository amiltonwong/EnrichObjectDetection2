% Load image
addpath('HoG');
addpath('HoG/features');
addpath('Util');

% Gives im, depth
demoRendering;
[im, depth ] = renderer.renderCrop();
if ~exist('Gamma','var')
  load('Statistics/sumGamma_N1_40_N2_40_sbin_4_nLevel_10_nImg_1601_napoli1_gamma.mat');
end

hog_cell_threshold = 1.5 * 10^0;
padding = 50;
n_cell_limits = [50 100 150 200 250 300 350 400];
lambda = 0.15;
time_per_case = zeros(1,numel(n_cell_limits));
residual_per_case = zeros(1,numel(n_cell_limits));
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

for caseIdx = 1:numel(n_cell_limits)
  n_cell_limit = n_cell_limits(caseIdx);
  
  
  tic
  HOGTemplate = esvm_initialize_goalsize_exemplar_ncell(paddedIm, bbox, n_cell_limit);

  %%%%%%%% WHO conversion using matrix decomposition

  sz = size(HOGTemplate);
  wHeight = sz(1);
  wWidth = sz(2);
  HOGDim = sz(3);

  sigmaDim = prod(sz);

  CircSigma = zeros(sigmaDim);
  Sigma = zeros(sigmaDim);

  for i = 1:wHeight
      for j = 1:wWidth
          rowIdx = i + wHeight * (j - 1); % sub2ind([wHeight, wWidth],i,j);
          for k = 1:wHeight
              for l = 1:wWidth
                  colIdx = k + wHeight * (l - 1); % sub2ind([wHeight, wWidth],k,l);
                  gammaRowIdx = abs(i - k) + 1;
                  gammaColIdx = abs(j - l) + 1;
                  Sigma((rowIdx-1)*HOGDim + 1:rowIdx * HOGDim, (colIdx-1)*HOGDim + 1:colIdx*HOGDim) = ...
                      Gamma((gammaRowIdx-1)*HOGDim + 1 : gammaRowIdx*HOGDim , (gammaColIdx - 1)*HOGDim + 1 : gammaColIdx*HOGDim);
              end
          end
      end
  end
  toc
  
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

  muSwapDim = permute(mu,[2 3 1]);

  %%%%%%%%%%%%%%%%%%%%%%%%%%
  % nonEmptyCells = (sum(HOGTemplate,3) > hog_cell_threshold);
  % centeredWs = bsxfun(@times,bsxfun(@minus,HOGTemplate,muSwapDim),nonEmptyCells);

  centeredWs = bsxfun(@minus,HOGTemplate,muSwapDim);
  %%%%%%%%%%%%%%%%%%%%%%%%%%

  centeredWs = permute(centeredWs,[3 1 2]); % [HOGDim, Nrow, Ncol] = HOGDim, N1, N2

  sigInvCenteredWsVec = R\(R'\centeredWs(:));
  sigInvCenteredWs = reshape(sigInvCenteredWsVec,[HOGDim, wHeight, wWidth]);
  WHOTemplate = permute(sigInvCenteredWs,[2,3,1]);
  
  time_per_case( caseIdx) = toc
  residual_per_case( caseIdx ) = norm(centeredWs(:) - Sigma * sigInvCenteredWsVec)
end

figure(1); plot(n_cell_limits, time_per_case,'-+'); title('Decomposition all cells'); xlabel('N Cells'); ylabel('seconds');
figure(2); plot(n_cell_limits, residual_per_case, '-+'); title('Decomposition all cells'); xlabel('N Cells'); ylabel('residuals');
figure(3); imagesc(HOGpicture(abs(WHOTemplate))); colorbar; title('Decomposition all cells');