% GPU Gamma Scramble Test
system('rm scrambleGammaToSigma.ptx');
system('nvcc -ptx scrambleGammaToSigma.cu');
k = parallel.gpu.CUDAKernel('scrambleGammaToSigma.ptx','scrambleGammaToSigma.cu');

N_Gamma_Dim = HOGDim * n_non_empty_cells;
N = 16;
k.ThreadBlockSize = [N, N, 1];
k.GridSize = [ceil(double(N_Gamma_Dim)/N), ceil(double(N_Gamma_Dim)/N), 1];

% scrambleGammaToSigma( float* Sigma, float* Gamma, int* nonEmptyRows, int* nonEmptyCols, int HOGDim, int nNonEmptyCells )
SigmaGPU = zeros(N_Gamma_Dim, N_Gamma_Dim, 'single', 'gpuArray');
GammaGPU = gpuArray(single(Gamma));
gammaDim = size(Gamma);
gammaDimGPU = gpuArray(int32(gammaDim(1)));

nonEmptyRowsGPU = gpuArray(int32(nonEmptyRows - 1));
nonEmptyColsGPU = gpuArray(int32(nonEmptyCols - 1));

HOGDimGPU = gpuArray(int32(HOGDim));
n_non_empty_cellsGPU = gpuArray(int32(n_non_empty_cells));


lambda = 0.015;

result = feval(k, SigmaGPU, GammaGPU, single(lambda), nonEmptyRowsGPU, nonEmptyColsGPU, gammaDimGPU, HOGDimGPU, n_non_empty_cellsGPU);
memResult = gather(result);
