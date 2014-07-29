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
n_cell_limits = [50 100 150 200 250 300 350 400 450 500];
% n_cell_limits = 200;
lambda = 0.01;

cg_time_per_case = zeros(1,numel(n_cell_limits));
cg_residual_per_case = zeros(1,numel(n_cell_limits));

CG_THREASHOLD = 10^-3;
CG_MAX_ITER = 60;
N_THREAD = 32;

%%%%%%%% Get HOG template

% create white background padding
paddedIm = padarray(im2double(im), [padding, padding, 0]);
paddedIm(:,1:padding,:) = 1;
paddedIm(:,end-padding+1 : end, :) = 1;
paddedIm(1:padding,:,:) = 1;
paddedIm(end-padding+1 : end, :, :) = 1;

% bounding box coordinate x1, y1, x2, y2
bbox = [1 1 size(im,2) size(im,1)] + padding;

scrambleKernel = parallel.gpu.CUDAKernel('scrambleGammaToSigma.ptx','scrambleGammaToSigma.cu');
scrambleKernel.ThreadBlockSize = [N_THREAD , N_THREAD , 1];

GammaGPU = gpuArray(single(Gamma));
gammaDim = size(Gamma);
  
for caseIdx = 1:numel(n_cell_limits)
  n_cell_limit = n_cell_limits(caseIdx);
  
  tic;
  
  [WHOTemplate_CG, ~, r_hist, residual] = WHOTemplateCG_GPU( im, scrambleKernel, mu, GammaGPU, gammaDim(1), n_cell_limit, lambda, padding, hog_cell_threshold, CG_THREASHOLD, CG_MAX_ITER, N_THREAD);
  
  cg_time_per_case(caseIdx) = toc

  cg_residual_per_case(caseIdx) = residual
end

figure(1); imagesc(HOGpicture(abs(WHOTemplate_CG))); colorbar; title('Conjugate Gradient nonzero cells');

figure(2); plot(n_cell_limits, cg_time_per_case,'-+'); title('Conjugate Gradient/Decomposition nonzero cells'); xlabel('N Cells'); ylabel('seconds');
legend('CG');

figure(3); plot(n_cell_limits, cg_residual_per_case, '-+'); title('Conjugate Gradient/Decomposition nonzero cells'); xlabel('N Cells'); ylabel('residuals');
legend('CG');

figure(4); semilogy(r_hist)