function [best_proposals]= dwot_binary_search_proposal_region(hog_region_pyramid, im_region, detectors, detector_table, param, im)

if ~isfield(param, 'renderer')
  error('Renderer not found');
end



n_proposal_region = numel(hog_region_pyramid);

n_batch = 1;
org_cell_limit = param.n_cell_limit;
% param.n_cell_limit = 200;

best_proposals = cell(1, n_proposal_region);
try
  for region_idx = 1:n_proposal_region
    % Initialize Chain
    template_idx = hog_region_pyramid{region_idx}.template_idx;
    best_state = struct('az', detectors{template_idx}.az,...
                    'el', detectors{template_idx}.el,...
                    'yaw', detectors{template_idx}.yaw,...
                    'fov', detectors{template_idx}.fov,...
                    'models_idx', cell(1, n_batch),... % there can be multiple models and renderings 
                    'template_size', cell(1, n_batch),...
                    'rendering_image', detectors{template_idx}.rendering_image,...
                    'image_bbox', hog_region_pyramid{region_idx}.image_bbox,...
                    'score', hog_region_pyramid{region_idx}.det_score);
    daz = param.azimuth_discretization;
    del = param.elevation_discretization;
    dyaw = param.yaw_discretization;
    dfov = param.fov_discretization;
    for depth_idx = 1:param.binary_search_max_depth
      [azs, els, yaws, fovs] = proposal_viewpoints(best_state.az,  best_state.el,  best_state.yaw,  best_state.fov,...
                          daz, del, dyaw, dfov,...
                          1, 1, 1, 0, param);
      fprintf('%d templates selected\n',numel(azs));
      detectors_subset = dwot_find_detector(detectors, detector_table, azs, els, yaws, fovs, [1], 'not_yet_supported', param);

      update_idx = mod( mcmc_iter - 1, 4) + 1;
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

      if 1
        figure(1);
        subplot(121);
        dwot_draw_overlap_detection(im, image_bbox, rendering_image, 5, 50, true);
        subplot(122);
        dwot_draw_overlap_detection(im, best_state(chain_idx).image_bbox, best_state(chain_idx).rendering_image, 5, 50, true);
        drawnow;
      end

      daz = daz / 2;
      del = del / 2;
      dyaw = dyaw / 2;
      dfov = dfov / 2;
    end
    
    best_proposals{region_idx} = best_state;
  end
catch e
  disp(e.message);
  renderer.delete();
  param.n_cell_limit = org_cell_limit;
  rethrow(e);
end
  renderer.delete();
param.n_cell_limit = org_cell_limit;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     return maximum score that got from the 
%     instant detector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [max_score, template, template_size, rendering_image, image_bbox] = dwot_detect_region(hog_pyramid, templates, param, im_region)
n_hog = numel(hog_pyramid.pyramid);
c = cell(1, n_hog);
max_score  = -inf;
max_level  = -1;
max_idx  = -1;
for level_idx = 1:n_hog
  c{level_idx} = fconvblasfloat(hog_pyramid.pyramid(level_idx).padded_hog, {WHOTemplate}, 1, 1);
  % c{level_idx} = dwot_conv(hog_pyramid.pyramid(level_idx).padded_hog, template, param);
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
  y_coord = y_coord + hog_pyramid.pyramid(max_level).padded_hog_bbox(2) - 2;
  x_coord = x_coord + hog_pyramid.pyramid(max_level).padded_hog_bbox(1) - 2;
  if param.computing_mode == 0
    [y1, x1] = dwot_hog_to_img_conv(y_coord, x_coord, param.sbin, hog_pyramid.pyramid(max_level).scale, param.detect_pyramid_padding);
    [y2, x2] = dwot_hog_to_img_conv(y_coord + template_size(1), x_coord + template_size(2), param.sbin, hog_pyramid.pyramid(max_level).scale, param.detect_pyramid_padding);
  else
    [y1, x1] = dwot_hog_to_img_fft(y_coord + hog_pyramid.template_size(1),...
                                    x_coord + hog_pyramid.template_size(2), ...
                                    template_size, param.sbin, hog_pyramid.pyramid(max_level).scale);
    [y2, x2] = dwot_hog_to_img_fft(y_coord + hog_pyramid.template_size(1) + template_size(1),...
                                    x_coord + hog_pyramid.template_size(2) + template_size(2),...
                                    template_size, param.sbin, hog_pyramid.pyramid(max_level).scale);
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
%     Proposal Viewpoints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [azs, els, yaws, fovs] = proposal_viewpoints(az,  el,  yaw,  fov,...
                                                      daz, del, dyaw, dfov,...
                                                      naz, nel, nyaw, nfov, param)

n_az_views = 2 * naz + 1;
n_el_views = 2 * nel + 1;
n_yaw_views = 2 * nyaw + 1;
n_fov_views = 2 * nfov + 1;

fovs = (-nfov:nfov)' * dfov + fov;
fovs = repmat(fovs, [n_az_views * n_el_views * n_yaw_views, 1]);
fovs = fovs(:);

yaws = (-nyaw:nyaw)' * dyaw + yaw;
yaws = repmat(yaws, [n_az_views * n_el_views, n_fov_views])';
yaws = yaws(:);

els = (-nel:nel)' * del + el;
els = repmat(els, [n_az_views, n_yaw_views * n_fov_views])';
els = els(:);

azs = (-naz:naz)' * dfov + fov;
azs = repmat(azs, [1, n_el_views * n_yaw_views * n_fov_views])';
azs = azs(:);