function [detectors, detector_name]= dwot_mcmc_proposal_region(Mu, Gamma, mesh_path, azs, els, yaws, fovs, n_cell_limit, lambda, visualize)

mesh_name = strrep(mesh_path, '/', '_');

detector_name = sprintf('%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d.mat',...
    mesh_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs));

  
if exist(detector_name,'file')
  load(detector_name);
else
  if exist('renderer','var')
    renderer.delete();
    clear renderer;
  end
  
  CG_THREASHOLD = 10^-4;
  CG_MAX_ITER = 60;
  N_THREAD = 32;
  
  hog_cell_threshold = 1.5;
  padding = 50;
  
  if ~exist('scrambleGammaToSigma.ptx','file')
    system('nvcc -ptx scrambleGammaToSigma.cu');
  end
  scrambleKernel = parallel.gpu.CUDAKernel('scrambleGammaToSigma.ptx','scrambleGammaToSigma.cu');
  scrambleKernel.ThreadBlockSize = [N_THREAD , N_THREAD , 1];

  GammaGPU = gpuArray(single(Gamma));
  gammaDim = size(Gamma);

  
  renderer = Renderer();
  if ~renderer.initialize([mesh_path], 700, 700, 0, 0, 0, 0, 25)
    error('fail to load model');
  end
  
  i = 1;
  detectors = cell(1,numel(azs) * numel(els) * numel(fovs));
  try
    for azIdx = 1:numel(azs)
      for elIdx = 1:numel(els)
        for yawIdx = 1:numel(yaws)
          for fovIdx = 1:numel(fovs)
            elGT = els(elIdx);
            azGT = azs(azIdx);
            yawGT = yaws(yawIdx);
            fovGT = fovs(fovIdx);
            tic
            renderer.setViewpoint(90-azGT,elGT,yawGT,0,fovGT);
            im = renderer.renderCrop();
            % [ WHOTemplate, HOGTemplate] = WHOTemplateDecompNonEmptyCell( im, Mu, Gamma, n_cell_limit, lambda, 50);
            % [ WHOTemplate, HOGTemplate] = WHOTemplateCG( im, Mu, Gamma, n_cell_limit, lambda, 50, 1.5, 10^-3, 100);
            [WHOTemplate, HOGTemplate] = WHOTemplateCG_GPU( im, scrambleKernel, Mu, GammaGPU, gammaDim(1), n_cell_limit, lambda, padding, hog_cell_threshold, CG_THREASHOLD, CG_MAX_ITER, N_THREAD);
            toc;

            detectors{i}.whow = WHOTemplate;
            % detectors{i}.hogw = HOGTemplate;
            detectors{i}.az = azGT;
            detectors{i}.el = elGT;
            detectors{i}.yaw = yawGT;
            detectors{i}.fov = fovGT;
            detectors{i}.rendering = im;
            detectors{i}.sz = size(WHOTemplate);
            % detectors{i}.whopic = HOGpicture(WHOTemplate);

            if visualize
              figure(1); subplot(131);
              imagesc(im); axis equal; axis tight;
              subplot(132);
              imagesc(HOGpicture(HOGTemplate)); axis equal; axis tight;
              subplot(133);
              imagesc(HOGpicture(WHOTemplate)); axis equal; axis tight;
              disp('press any button to continue');
              waitforbuttonpress;
            end
            i = i + 1;    
          end
        end
      end
    end
  catch e
    disp(e.message);
  end
  renderer.delete();
end