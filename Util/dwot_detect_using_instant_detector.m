%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     return maximum score that got from the 
%     instant detector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [max_score, template, template_size, paddedIm, paddedDepth, image_bbox] = ...
    dwot_detect_using_instant_detector(renderer, hog_pyramid, az, el, yaw, fov, models_idx, param, im_region)
renderer.setViewpoint(az,el,yaw,0,fov);
% renderer.setModelIndex(models_idx);
[rendering_image, rendering_depth] = renderer.renderCrop();

[template, ~, scale ] = WHOTemplateCG_CUDA( rendering_image, param);

padding = round(param.rendering_sbin / scale / 2);

size_rendering = size(rendering_image);
paddedIm = 255 * ones(size_rendering + [2 * padding, 2 * padding, 0],'uint8');
paddedIm(padding+1:padding+size_rendering(1), padding+1:padding+size_rendering(2),:) = rendering_image;
paddedDepth = zeros(size_rendering(1:2) + [2 * padding, 2 * padding]);
paddedDepth(padding+1:padding+size_rendering(1), padding+1:padding+size_rendering(2)) = rendering_depth;

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
    [y1, x1] = dwot_hog_to_img_conv(y_coord , x_coord, param.sbin, hog_pyramid.pyramid(max_level).scale, param.detect_pyramid_padding);
    [y2, x2] = dwot_hog_to_img_conv(y_coord + template_size(1), x_coord + template_size(2), param.sbin, hog_pyramid.pyramid(max_level).scale, param.detect_pyramid_padding);
  elseif param.computing_mode == 1
          % detection
%     [y1, x1] = dwot_hog_to_img_fft(y_coord - 0.5, x_coord - 0.5, sz{templateIdx}, sbin, scale);
%     [y2, x2] = dwot_hog_to_img_fft(y_coord + sz{templateIdx}(1) + 0.5 , x_coord + sz{templateIdx}(2) + 0.5, sz{templateIdx}, sbin, scale);    
    [y1, x1] = dwot_hog_to_img_fft(y_coord - 0.5, x_coord - 0.5, [1 1], param.sbin, hog_pyramid.pyramid(max_level).scale);
    [y2, x2] = dwot_hog_to_img_fft(y_coord + template_size(1) + 0.5,...
                                    x_coord + template_size(2) + 0.5,...
                                    [ 1 1 ], param.sbin, hog_pyramid.pyramid(max_level).scale);
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


