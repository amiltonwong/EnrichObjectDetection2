
hog_cell_threshold = 1.5 * 10^0;
padding = 50;
n_cell_limits = [50 100 150 200 250 300 350 400];
lambda = 0.15;
time_per_case = zeros(1,numel(n_cell_limits));
residual_per_case = zeros(1,numel(n_cell_limits));
%%%%%%%% Get HOG template

param.scramble_gamma_to_sigma_file = './scrambleGammaToSigma';
if ~exist([param.scramble_gamma_to_sigma_file  '.ptx'],'file')
    system(['nvcc -ptx ' param.scramble_gamma_to_sigma_file '.cu']);
end
scramble_kernel                  = parallel.gpu.CUDAKernel([param.scramble_gamma_to_sigma_file '.ptx'],[param.scramble_gamma_to_sigma_file '.cu']);
scramble_kernel.ThreadBlockSize  = [param.N_THREAD_H , param.N_THREAD_W , 1];
param.scramble_kernel = scramble_kernel;

% create white background padding
paddedIm = padarray(im2double(im), [padding, padding, 0]);
paddedIm(:,1:padding,:) = 1;
paddedIm(:,end-padding+1 : end, :) = 1;
paddedIm(1:padding,:,:) = 1;
paddedIm(end-padding+1 : end, :, :) = 1;

% bounding box coordinate x1, y1, x2, y2
bbox = [1 1 size(im,2) size(im,1)] + padding;

% TODO replace it
    gammaDim            = param.hog_gamma_dim;
for caseIdx = 1:numel(n_cell_limits)
  n_cell_limit = n_cell_limits(caseIdx);

  HOGTemplate = esvm_initialize_goalsize_exemplar_ncell(paddedIm, bbox, n_cell_limit);

  %%%%%%%% WHO conversion using matrix decomposition

  sz = size(HOGTemplate);
  wHeight = sz(1);
  wWidth = sz(2);
  HOGDim = sz(3);

  
  
    nonEmptyCells = (sum(HOGTemplate,3) > hog_cell_threshold);
    idxNonEmptyCells = find(nonEmptyCells);

    muSwapDim = permute(Mu,[2 3 1]);
    centeredHOG = bsxfun(@minus, HOGTemplate, muSwapDim);
    permHOG = permute(centeredHOG,[3 1 2]); % [HOGDim, Nrow, Ncol] = HOGDim, N1, N2
    onlyNonEmptyIdx = cell2mat(arrayfun(@(x) x + (1:HOGDim)', HOGDim * (idxNonEmptyCells - 1),'UniformOutput',false));
    nonEmptyHOG = permHOG(onlyNonEmptyIdx);

    [nonEmptyRows,nonEmptyCols] = ind2sub([wHeight, wWidth], idxNonEmptyCells);
    nonEmptyRows = int32(nonEmptyRows);
    nonEmptyCols = int32(nonEmptyCols);

    
    n_non_empty_cells = int32(numel(nonEmptyRows));

    tic
    sigmaDim = n_non_empty_cells * HOGDim;
    for repeat = 1:100
        SigmaGPU = zeros(sigmaDim, sigmaDim, 'single', 'gpuArray');
        param.scramble_kernel.GridSize = [ceil(double(sigmaDim)/param.N_THREAD_H ), ceil(double(sigmaDim)/param.N_THREAD_W ), 1];

        nonEmptyRowsGPU = gpuArray(nonEmptyRows - 1);
        nonEmptyColsGPU = gpuArray(nonEmptyCols - 1);

        AGPU = feval(param.scramble_kernel, SigmaGPU, param.hog_gamma_gpu, single(lambda), nonEmptyRowsGPU, nonEmptyColsGPU, gammaDim(1), HOGDim, n_non_empty_cells);

        A_kernel = gather(AGPU);
    end
    gpuTime(caseIdx)=toc
    
%   sigmaDim = prod(sz);

tic

    for repeat = 1:100

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
    end
  cpuTime(caseIdx) = toc
  
  
end

set(0,'DefaultAxesFontName', 'helvetica')
set(0,'DefaultAxesFontSize', 20)

% Change default text fonts.
set(0,'DefaultTextFontname', 'helvetica')
set(0,'DefaultTextFontSize', 20)

subplot(121);
plot(n_cell_limits, gpuTime/100,'b-+');  xlabel('Number of HOG Cells'); ylabel('Seconds');
hold on;
plot(n_cell_limits, cpuTime/100, 'r-*');
legend('GPU','CPU');

subplot(122);
semilogy(n_cell_limits, gpuTime/100,'b-+');  xlabel('Number of HOG Cells'); ylabel('Seconds');
hold on;
semilogy(n_cell_limits, cpuTime/100, 'r-*');
legend('GPU','CPU');
