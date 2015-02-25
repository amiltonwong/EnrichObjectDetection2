function resultIm = dwot_draw_overlap_rendering(im, bbsNMS, detectors, max_n_draw_box, draw_padding, box_text, rendering_image_weight, color_range, text_mode)

if nargin < 4
  max_n_draw_box = 5;
end

if nargin  < 5
  draw_padding = 25;
end

if nargin < 6
  box_text = false;
end

if nargin < 7
  rendering_image_weight = ones(1,2)/2;
end

if nargin < 8
  color_range = false;
end

if nargin < 9
    text_mode = 0;
end

resultIm = pad_image(im2double(im), draw_padding, 1);
% resultIm = paddedIm;

NDrawBox = min(size(bbsNMS, 1),max_n_draw_box);

% Create overlap image
clipBBox = zeros(NDrawBox,4);
for bbsIdx = NDrawBox:-1:1
    template_idx = bbsNMS(bbsIdx,11);
    
    rendering  = im2double(detectors{template_idx}.rendering_image);
    depth_mask =           detectors{template_idx}.rendering_depth;

    box_position = round(bbsNMS(bbsIdx, 1:4)) + draw_padding;
    
    [resultIm, clip_bnd] = dwot_draw_overlay_rendering(resultIm, box_position, rendering, depth_mask, rendering_image_weight);
    clipBBox(bbsIdx,:) = clip_bnd;
end

if box_text
  cla;
  imagesc(resultIm);
  
  % Draw bounding box.
  if ~color_range
    color_map = hot(NDrawBox);
  else
    n_color = numel(color_range);
    color_map = jet(n_color);
  end
  
  for bbsIdx = NDrawBox:-1:1
    curr_color = dwot_color_from_range(bbsNMS(bbsIdx,12), color_range, color_map);
    box_position = clipBBox(bbsIdx, 1:4);
    
    % if detector id available (positive number), print it
    if bbsNMS(bbsIdx,11) > 0 && (text_mode == 0)
      box_text = sprintf(' s:%0.2f o:%0.2f t:%d',bbsNMS(bbsIdx,12),bbsNMS(bbsIdx,9),bbsNMS(bbsIdx,11));
    elseif (text_mode == 1)
      box_text = sprintf(' s:%0.2f o:%0.2f ',bbsNMS(bbsIdx,12),bbsNMS(bbsIdx,9));
    elseif (text_mode == 2)
      box_text = sprintf(' s:%0.2f ',bbsNMS(bbsIdx,12));
    elseif (text_mode == 3)
      box_text = sprintf(' s:%0.2f a:%0.0f o:%0.2f',bbsNMS(bbsIdx,12), bbsNMS(bbsIdx,10), bbsNMS(bbsIdx,9));
    end
    
    dwot_visualize_bounding_box_and_text(box_position, box_text, curr_color);    
  end
  axis equal;
  axis tight;
  axis off;
  drawnow;
end
