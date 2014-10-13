function [best_proposals]= dwot_mcmc_proposal_region(renderer, hog_region_pyramid, im_region, detectors, param, im, visualize)

if nargin <7 
    visualize = false;
end

n_proposal_region = numel(hog_region_pyramid);

n_batch = 1;
org_cell_limit = param.n_cell_limit;
% param.n_cell_limit = 200;

best_proposals = cell(1, n_proposal_region);
% try
  for region_idx = 1:n_proposal_region
    % Initialize Chain
    template_idx = hog_region_pyramid{region_idx}.template_idx;
    best_state = struct('x', [detectors{template_idx}.az, detectors{template_idx}.el, detectors{template_idx}.yaw, detectors{template_idx}.fov],...
                    'models_idx', detectors{template_idx}.model_index,... % there can be multiple models and renderings 
                    'template_size', cell(1, n_batch),...
                    'rendering_image', detectors{template_idx}.rendering_image,...
                    'image_bbox', hog_region_pyramid{region_idx}.image_bbox,...
                    'score', hog_region_pyramid{region_idx}.det_score);

    current_state = struct('x', [detectors{template_idx}.az, detectors{template_idx}.el, detectors{template_idx}.yaw, detectors{template_idx}.fov],...
                    'models_idx', detectors{template_idx}.model_index,... % for multiple model
                    'score', hog_region_pyramid{region_idx}.det_score);

    % Run Chain using Gibbs sampling
    for mcmc_iter = 1:param.mcmc_max_iter
      update_idx = mod( mcmc_iter - 1, 4) + 1;
      for chain_idx = 1:n_batch
        proposal_x = current_state(chain_idx).x;
        proposal_x(update_idx) = proposal_x(update_idx) + 5 * randn(1);
        models_idx = current_state(chain_idx).models_idx;
        [max_score, template, template_size, rendering_image, image_bbox] = dwot_detect_using_instant_detector(renderer, hog_region_pyramid{region_idx}, proposal_x(1), proposal_x(2), proposal_x(3), proposal_x(4), models_idx, param, im_region{region_idx});
        if max_score > best_state(chain_idx).score
          fprintf(sprintf('region %d iter %d : %f\n',region_idx, mcmc_iter, max_score));
          best_state(chain_idx).score = max_score;
          best_state(chain_idx).template_size = template_size;
          best_state(chain_idx).x = proposal_x;
          best_state(chain_idx).image_bbox = image_bbox;
          best_state(chain_idx).rendering_image = rendering_image;
        end
        
        if visualize
          subplot(121);
          image_bbox(:,12) = max_score;
          if ~isempty(rendering_image) && max_score > -inf
            dwot_draw_overlap_detection(im, image_bbox, rendering_image, 5, 50, true);
          end
          
          subplot(122);
          bestBox = best_state(chain_idx).image_bbox;
          bestBox(:,12) = best_state(chain_idx).score;
          dwot_draw_overlap_detection(im, bestBox, best_state(chain_idx).rendering_image, 5, 50, true);
          drawnow;
        end
        % Metropolis Hastings
        acc = min(1, probability_from_score((max_score - current_state(chain_idx).score)/param.n_cell_limit * 100));
        if rand(1) < acc
          fprintf('.');
          current_state(chain_idx).x = proposal_x;
          current_state(chain_idx).score = max_score;
        end
      end
    end
    
    best_proposals{region_idx} = best_state;
  end
% catch e
%   disp(e.message);
%   param.n_cell_limit = org_cell_limit;
%   rethrow(e);
% end
param.n_cell_limit = org_cell_limit;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     return maximum score that got from the 
%     instant detector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [max_score, template, template_size, rendering_image, image_bbox] = dwot_detect_using_instant_detector(renderer, hog_pyramid, az, el, yaw, fov, models_idx, param, im_region)
renderer.setViewpoint(az,el,yaw,0,fov);
renderer.setModelIndex(models_idx);
rendering_image = renderer.renderCrop();
template = WHOTemplateCG_CUDA( rendering_image, param);
template_size = size(template);
n_hog = numel(hog_pyramid.pyramid);
c = cell(1, n_hog);
max_score  = -inf;
max_level  = -1;
max_idx  = -1;
for level_idx = 1:n_hog
  % c{level_idx} = fconvblasfloat(hog_pyramid.pyramid(level_idx).padded_hog, {WHOTemplate}, 1, 1);
  c{level_idx} = dwot_conv(hog_pyramid.pyramid(level_idx).padded_hog, template, param);
  [max_c, max_c_idx] = max(c{level_idx}(:));
  if max_score < max_c;
    max_score = max_c;
    max_level = level_idx;
    max_idx   = max_c_idx;
  end
end

if max_level == -1
  image_bbox = zeros(4,1);
  return;
else
  data_size = size(c{max_level});
  [y_coord, x_coord] = ind2sub(data_size(1:2), max_idx);
  y_coord = y_coord + hog_pyramid.pyramid(max_level).padded_hog_bbox(2) - 1;
  x_coord = x_coord + hog_pyramid.pyramid(max_level).padded_hog_bbox(1) - 1;

  if param.computing_mode == 0
    [y1, x1] = dwot_hog_to_img_conv(y_coord, x_coord, param.sbin, hog_pyramid.pyramid(max_level).scale, param.detect_pyramid_padding);
    [y2, x2] = dwot_hog_to_img_conv(y_coord + template_size(1), x_coord + template_size(2), param.sbin, hog_pyramid.pyramid(max_level).scale, param.detect_pyramid_padding);
  elseif param.computing_mode == 1
    [y1, x1] = dwot_hog_to_img_fft(y_coord, x_coord, [1 1], param.sbin, hog_pyramid.pyramid(max_level).scale);
    [y2, x2] = dwot_hog_to_img_fft(y_coord+ template_size(1),...
                                    x_coord + template_size(2),...
                                    [1 1], param.sbin, hog_pyramid.pyramid(max_level).scale);
  else
    error('Computing mode not supported');
  end
  image_bbox = [x1 y1, x2, y2];
  max_score = double(max_score);
  % fprintf(['Score : ' num2str(max_score) '\n']);

  if 0
    subplot(221);
    imagesc(im_region);
  %   subplot(222);
  %   imagesc(HOGpicture(WHOTemplate));
    subplot(223);
    imagesc(rendering_image);
    drawnow
  %   for level_idx = 1:n_hog
  %     subplot(224);
  %     imagesc(c{level_idx}{1});
  %     colorbar;
  %     waitforbuttonpress
  %   end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     unnormalized probability from score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p = probability_from_score(score)
p = exp(score);
