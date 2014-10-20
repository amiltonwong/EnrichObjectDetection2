function [detectors]= dwot_make_detectors_grid(renderer, azs, els, yaws, fovs, model_indexes, model_class, param, visualize)

if nargin < 7
  visualize = false;
end

%  Container class for fast query. Hash table k
% param.detector_table = containers.Map;
% if ~isfield(param, 'renderer')
%   renderer = Renderer();
%   if ~renderer.initialize([mesh_path], 700, 700, 0, 0, 0, 0, 25)
%     error('fail to load model');
%   end
%   param.renderer = renderer;
% end

i = 1;
start_time = tic;
detectors = cell(1,numel(azs) * numel(els) * numel(fovs));
for model_index = model_indexes
  renderer.setModelIndex(model_index);
  for azIdx = 1:numel(azs)
    for elIdx = 1:numel(els)
      for yawIdx = 1:numel(yaws)
        for fovIdx = 1:numel(fovs)
          elGT = els(elIdx);
          azGT = azs(azIdx);
          yawGT = yaws(yawIdx);
          fovGT = fovs(fovIdx);

          detector = dwot_get_detector(renderer, azGT, elGT, yawGT, fovGT, model_index, 'not_supported_model_class', param);
          
          detectors{i} = detector;
          % param.detector_table( dwot_detector_key(azGT, elGT, yawGT, fovGT) ) = i;

          if visualize
            subplot(121);
            imagesc(detector.rendering_image); axis equal; axis tight;
            % subplot(132);
            % imagesc(HOGpicture(HOGTemplate)); axis equal; axis tight;
            subplot(122);
            imagesc(HOGpicture(detector.whow)); axis equal; axis tight;
            drawnow;
%            disp('press any button to continue');
%            waitforbuttonpress;
          end
          i = i + 1;
                    
          if mod(i,20)==1
              fprintf('average time : %0.2f sec\n', toc(start_time)/i);
          else
              fprintf('.');
          end
        end
      end
    end
  end
end
