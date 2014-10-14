function [best_proposals, detectors, detector_table]= dwot_binary_search_proposal_region(hog_region_pyramid, im_region, detectors, detector_table, renderer, param, im)

n_proposal_region = numel(hog_region_pyramid);

n_batch = 1;
org_cell_limit = param.n_cell_limit;
% param.n_cell_limit = 200;

best_proposals = cell(1, n_proposal_region);
for region_idx = 1:n_proposal_region
  % Initialize Chain
  template_idx = hog_region_pyramid{region_idx}.template_idx;
  best_state = struct('az', detectors{template_idx}.az,...
                  'el', detectors{template_idx}.el,...
                  'yaw', detectors{template_idx}.yaw,...
                  'fov', detectors{template_idx}.fov,...
                  'models_idx', detectors{template_idx}.model_index,... % there can be multiple models and renderings 
                  'template_size', detectors{template_idx}.sz,...
                  'rendering_image', detectors{template_idx}.rendering_image,...
                  'image_bbox', hog_region_pyramid{region_idx}.image_bbox,...
                  'score', hog_region_pyramid{region_idx}.det_score);
  daz = param.azimuth_discretization;
  del = param.elevation_discretization;
  dyaw = param.yaw_discretization;
  dfov = param.fov_discretization;
  
  daz = daz / 2;
  del = del / 2;
  dyaw = dyaw / 2;
  dfov = dfov / 2;
  for depth_idx = 1:param.binary_search_max_depth
    [azs, els, yaws, fovs] = dwot_proposal_viewpoint_grid(best_state.az,  best_state.el,  best_state.yaw,  best_state.fov,...
                        daz, del, dyaw, dfov,...
                        1, 1, 1, 1, param);
    [detectors_subset, detector_subset_indexes] = dwot_find_detector(detectors, detector_table, azs, els, yaws, fovs, [1], 'not_yet_supported', param);
    % Create detectors that was not found in the pool of detectors
    empty_detectors_indexes = find(cellfun(@(x) isempty(x), detectors_subset));
    detectors_subset(empty_detectors_indexes) = dwot_make_detectors(renderer, azs(empty_detectors_indexes),...
              els(empty_detectors_indexes),...
              yaws(empty_detectors_indexes),...
              fovs(empty_detectors_indexes),...
              param);
    detector_subset_indexes(empty_detectors_indexes) = numel(detectors) + (1:numel(empty_detectors_indexes));

    fprintf('%d/%d templates created\n', numel(empty_detectors_indexes), numel(azs));
    % Add detectors to the variable
    [detectors, detector_table]= dwot_make_table_from_detectors([detectors, detectors_subset(empty_detectors_indexes)], detector_table);
    [max_score, max_template_subset_index, image_bbox] = dwot_detect_region(hog_region_pyramid{region_idx}, cellfun(@(x) x.whow, detectors_subset,'UniformOutput',false), param, im_region{region_idx});
    max_detector_index = detector_subset_indexes(max_template_subset_index);
    if max_score > best_state.score
      fprintf(sprintf('region %d depth %d : %f\n',region_idx, depth_idx, max_score));
      best_state.score = max_score;
      best_state.template_index = max_detector_index;
      best_state.image_bbox = image_bbox;
      best_state.rendering_image = detectors{max_detector_index}.rendering_image;
    end

    if 1
      figure(1);
      subplot(121);
      image_bbox_orig = hog_region_pyramid{region_idx}.image_bbox;
      image_bbox_orig(:,12) = hog_region_pyramid{region_idx}.det_score;
      dwot_draw_overlap_detection(im, image_bbox_orig, detectors{hog_region_pyramid{region_idx}.template_idx}.rendering_image, 5, 50, true);
      subplot(122);
      best_box = best_state.image_bbox;
      best_box(:,12) = best_state.score;
      dwot_draw_overlap_detection(im, best_box, best_state.rendering_image, 5, 50, true);
      drawnow;
    end

    daz = daz / 2;
    del = del / 2;
    dyaw = dyaw / 2;
    dfov = dfov / 2;
  end

  best_proposals{region_idx} = best_state;
end

param.n_cell_limit = org_cell_limit;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     return maximum score that got from the 
%     instant detector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [max_score, max_template_idx, image_bbox] = dwot_detect_region(hog_pyramid, templates, param, im_region)
n_hog = numel(hog_pyramid.pyramid);
c = cell(1, n_hog);
max_score  = -inf;
max_level  = -1;
max_coord_idx  = -1;
n_template = numel(templates);
for level_idx = 1:n_hog
  c{level_idx} = fconvblasfloat(hog_pyramid.pyramid(level_idx).padded_hog, templates, 1, n_template);
  % c{level_idx} = dwot_conv(hog_pyramid.pyramid(level_idx).padded_hog, template, param);
  for template_index = 1:n_template
    [max_c, max_c_idx] = max(c{level_idx}{template_index}(:));
    if max_score < max_c;
      max_score = max_c;
      max_level = level_idx;
      max_coord_idx   = max_c_idx;
      max_template_idx = template_index;
    end
  end
end

% if max_level == -1
%   image_bbox = zeros(4,1);
%   return;
% else
%   data_size = size(c{max_level}{max_template_idx});
%   [y_coord, x_coord] = ind2sub(data_size(1:2), max_coord_idx);
%   y_coord = y_coord + hog_pyramid.pyramid(max_level).padded_hog_bbox(2) - 2;
%   x_coord = x_coord + hog_pyramid.pyramid(max_level).padded_hog_bbox(1) - 2;
%   if param.computing_mode == 0
%     [y1, x1] = dwot_hog_to_img_conv(y_coord, x_coord, param.sbin, hog_pyramid.pyramid(max_level).scale, param.detect_pyramid_padding);
%     [y2, x2] = dwot_hog_to_img_conv(y_coord + template_size(1), x_coord + template_size(2), param.sbin, hog_pyramid.pyramid(max_level).scale, param.detect_pyramid_padding);
%   else
%     [y1, x1] = dwot_hog_to_img_fft(y_coord + hog_pyramid.template_size(1),...
%                                     x_coord + hog_pyramid.template_size(2), ...
%                                     template_size, param.sbin, hog_pyramid.pyramid(max_level).scale);
%     [y2, x2] = dwot_hog_to_img_fft(y_coord + hog_pyramid.template_size(1) + template_size(1),...
%                                     x_coord + hog_pyramid.template_size(2) + template_size(2),...
%                                     template_size, param.sbin, hog_pyramid.pyramid(max_level).scale);
%   end
%   image_bbox = [x1 y1, x2, y2];
%   max_score = double(max_score);
%   % fprintf(['Score : ' num2str(max_score) '\n']);
% 
%   if 0
%     subplot(221);
%     imagesc(im_region);
%   %   subplot(222);
%   %   imagesc(HOGpicture(WHOTemplate));
%     subplot(223);
%     imagesc(rendering_image);
%     drawnow
%   %   for level_idx = 1:n_hog
%   %     subplot(224);
%   %     imagesc(c{level_idx}{1});
%   %     colorbar;
%   %     waitforbuttonpress
%   %   end
%   end
% end

if max_level == -1
  image_bbox = zeros(4,1);
  return;
else
  template = templates{max_template_idx};
  template_size = size(template);
  data_size = size(c{max_level}{max_template_idx});
  [y_coord, x_coord] = ind2sub(data_size(1:2), max_coord_idx);
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


