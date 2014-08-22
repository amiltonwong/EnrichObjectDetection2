function [detectors]= dwot_bfgs_proposal_region(hog_region_pyramid, im_region, detectors, param, im)

renderer = Renderer();
if ~renderer.initialize([param.models_path{1} '.3ds'], 700, 700, 0, 0, 0, 0, 25)
  error('fail to load model');
end

n_proposal_region = numel(hog_region_pyramid);

% best_state = struct('az', 0,...
%                     'el', 0,...
%                     'yaw', 0,...
%                     'fov', 0,...
%                     'bbox', zeros(1,4),...
%                     'renderings', cell(1),...
%                     'models', cell(1),... % there can be multiple models and renderings
%                     'models_idx', cell(1),...
%                     'energy', -inf);


options = optimoptions(@fminunc,'Algorithm','quasi-newton','MaxIter',5,'FinDiffRelStep',[0.05;0.05;0.05;0.05],'Display','none');
org_cell_limit = param.n_cell_limit;
% param.n_cell_limit = 200;

try
  for region_idx = 1:n_proposal_region
    template_idx = hog_region_pyramid{region_idx}.template_idx;
    current_state = struct('az', detectors{template_idx}.az,...
                    'el', detectors{template_idx}.el,...
                    'yaw', detectors{template_idx}.yaw,...
                    'fov', detectors{template_idx}.fov,...
                    'models_idx', cell(1),...
                    'energy', hog_region_pyramid{region_idx}.det_score);
    models_idx = [];
    compute_score = @(x) -dwot_detect_using_instant_detector(renderer,...
                                                hog_region_pyramid{region_idx},...
                                                x(1), x(2), x(3), x(4),...
                                                models_idx,...
                                                param,...
                                                im_region{region_idx});
    [X(:,region_idx), F(region_idx), exitflag] = fminunc(@(x)compute_score(x),...
                                        [current_state.az, current_state.el, current_state.yaw, current_state.fov]...
                                        ,options);
    if 0
      figure(1); subplot(131);
      imagesc(rendering_image); axis equal; axis tight;
      subplot(132);
      imagesc(HOGpicture(HOGTemplate)); axis equal; axis tight;
      subplot(133);
      imagesc(HOGpicture(WHOTemplate)); axis equal; axis tight;
      disp('press any button to continue');
      waitforbuttonpress;
    end
  end
catch e
  disp(e.message);
  renderer.delete();
  param.n_cell_limit = org_cell_limit;
  rethrow(e);
end
renderer.delete();
param.n_cell_limit = org_cell_limit;


function max_score = dwot_detect_using_instant_detector(renderer, hog_pyramid, az, el, yaw, fov, models_idx, param, im_region)
renderer.setViewpoint(90-az,el,yaw,0,fov);
rendering_image = renderer.renderCrop();
WHOTemplate = WHOTemplateCG_GPU( rendering_image, param);
n_hog = numel(hog_pyramid.pyramid);
c = cell(1, n_hog);
max_score = -inf;
for level_idx = 1:n_hog
  c{level_idx} = fconvblasfloat(hog_pyramid.pyramid(level_idx).padded_hog, {WHOTemplate}, 1, 1);
  max_c = max(c{level_idx}{1}(:));
  if max_score < max_c;
    max_score = max_c;
  end
end

if 1
  subplot(221);
  imagesc(im_region);
%   subplot(222);
%   imagesc(HOGpicture(WHOTemplate));
  subplot(223);
  imagesc(rendering_image);
  title(['Score : ' num2str(max_score)]);
%   for level_idx = 1:n_hog
%     subplot(224);
%     imagesc(c{level_idx}{1});
%     colorbar;
%     waitforbuttonpress
%   end
  waitforbuttonpress
end
fprintf(['Score : ' num2str(max_score) '\n']);
max_score = double(max_score);
