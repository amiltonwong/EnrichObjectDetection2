function resultIm = dwot_draw_overlap_rendering(im, bbsNMS, detectors, max_n_draw_box, draw_padding, box_text, drawing_weights, color_range, text_mode)

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
  drawing_weights = ones(1,2)/2;
end

if nargin < 8
  color_range = false;
end

if nargin < 9
    text_mode = 0;
end

paddedIm = pad_image(im2double(im), draw_padding, 1);
resultIm = paddedIm;

NDrawBox = min(size(bbsNMS, 1),max_n_draw_box);

% Create overlap image
clipBBox = zeros(NDrawBox,4);
for bbsIdx = NDrawBox:-1:1
    template_idx = bbsNMS(bbsIdx,11);
    
    rendering  = im2double(detectors{template_idx}.rendering_image);
    depth_mask =           detectors{template_idx}.rendering_depth;

    box_position = round(bbsNMS(bbsIdx, 1:4)) + draw_padding;
    bboxWidth  = box_position(3) - box_position(1);
    bboxHeight = box_position(4) - box_position(2);
    szPadIm  = size(paddedIm);

    clip_bnd = [ min(box_position(1),szPadIm(2)),...
          min(box_position(2), szPadIm(1)),...
          min(box_position(3), szPadIm(2)),...
          min(box_position(4), szPadIm(1))];
    clip_bnd = [max(clip_bnd(1),1),...
          max(clip_bnd(2),1),...
          max(clip_bnd(3),1),...
          max(clip_bnd(4),1)];

    renderingSz = size(rendering);
    cropRegion = round((box_position - clip_bnd) ./ [bboxWidth, bboxHeight, bboxWidth, bboxHeight] .*...
                        [renderingSz(2), renderingSz(1), renderingSz(2), renderingSz(1)]);
	crop_rendering = rendering(...
              (1 - cropRegion(2)):(end - cropRegion(4)),...
              (1 - cropRegion(1)):(end - cropRegion(3)),:);
%     resizeRendering = imresize(crop_rendering,...
%               [clip_bnd(4) - clip_bnd(2) + 1, clip_bnd(3) - clip_bnd(1) + 1]);
%     resizeRendering(resizeRendering(:)>1)=1;
%     resizeRendering(resizeRendering(:)<0)=0;

    % Crop out the result image and resize it to the rendering and transfer
    % the rendering. This way, we won't see any noisy white pixels after
    % the image resizing. Image resizing uses interpolation which is
    % inexact for our rendering.
    curr_depth = depth_mask( (1 - cropRegion(2)):(end - cropRegion(4)),...
                           (1 - cropRegion(1)):(end - cropRegion(3)));
    curr_depth_mask = curr_depth > 0;
    curr_depth_mask = repmat(curr_depth_mask, [1, 1, 3]);

    resultImCrop = resultIm( clip_bnd(2):clip_bnd(4), clip_bnd(1):clip_bnd(3), :);
    size_cropped_rendering = size(curr_depth);
    resizeResultImCrop = imresize(resultImCrop,[size_cropped_rendering(1), size_cropped_rendering(2)]);
    resizeResultImCrop(curr_depth_mask) = crop_rendering(curr_depth_mask);
    rendering_transferred_result_im = resizeResultImCrop;
    rendering_transferred_result_im = imresize(rendering_transferred_result_im,...
                        [clip_bnd(4)-clip_bnd(2)+1, clip_bnd(3)-clip_bnd(1)+1]);
    rendering_transferred_result_im(rendering_transferred_result_im(:)>1)=1;
    rendering_transferred_result_im(rendering_transferred_result_im(:)<0)=0;
    % Done resizing and transfering rendering.
    
    
    resizeDepth = imresize(curr_depth,...
                    [clip_bnd(4)-clip_bnd(2)+1, clip_bnd(3)-clip_bnd(1)+1]);
    
    % Interpolation might introduce artifacts
	resizeDepth(resizeDepth(:)>1)=1;
    resizeDepth(resizeDepth(:)<0)=0;

    % Conver the depth mask to logical and use it for mask
    depth_mask = resizeDepth > 0;
    depth_mask = repmat(depth_mask, [1, 1, 3]);
    
    % resizeRendering = resizeRendering(1:(clip_bnd(4) - clip_bnd(2) + 1), 1:(clip_bnd(3) - clip_bnd(1) + 1), :);
    % bndIm = paddedIm( clip_bnd(2):clip_bnd(4), clip_bnd(1):clip_bnd(3), :);
%    resultImCrop = resultIm( clip_bnd(2):clip_bnd(4), clip_bnd(1):clip_bnd(3), :);
%    blendIm =             resultImCrop      * drawing_weights(3)...
%                       + bndIm             * drawing_weights(1);
    blendIm = resultImCrop;
    blendIm(depth_mask) = resultImCrop(depth_mask) * drawing_weights(1) + rendering_transferred_result_im(depth_mask) * drawing_weights(2);
  
    resultIm(clip_bnd(2):clip_bnd(4), clip_bnd(1):clip_bnd(3),:) = blendIm;
    clipBBox(bbsIdx,:) = clip_bnd;
end

if box_text
  cla;
  imagesc(resultIm);
  
%   if size(bbsNMS,2) < 12
%     bbsNMS(:,12) = 0;
%   end
  
  % Draw bounding box.
  if ~color_range
    color_map = hot(NDrawBox);
  else
    n_color = numel(color_range);
    color_map = jet(n_color);
  end
  
  for bbsIdx = NDrawBox:-1:1
    if ~color_range
      curr_color = color_map( bbsIdx ,:);
    else
      [~, color_idx] = histc(bbsNMS(bbsIdx,12), color_range);
      curr_color = color_map(color_idx, :);
    end
    
    box_position = clipBBox(bbsIdx, 1:4) + [0 0 -clipBBox(bbsIdx, 1:2)];
    
    % if detector id available (positive number), print it
    if bbsNMS(bbsIdx,11) > 0 && (text_mode == 0)
      box_text = sprintf(' s:%0.2f o:%0.2f t:%d',bbsNMS(bbsIdx,12),bbsNMS(bbsIdx,9),bbsNMS(bbsIdx,11));
    elseif (text_mode == 0)
      box_text = sprintf(' s:%0.2f o:%0.2f ',bbsNMS(bbsIdx,12),bbsNMS(bbsIdx,9));
    elseif (text_mode == 1)
      box_text = sprintf(' s:%0.2f ',bbsNMS(bbsIdx,12));
    end
    
    rectangle('position', box_position,'edgecolor',[0.5 0.5 0.5],'LineWidth',3);
    rectangle('position', box_position,'edgecolor',curr_color,'LineWidth',1);
    text(box_position(1) + 1 , box_position(2), box_text, 'BackgroundColor', curr_color,'EdgeColor',[0.5 0.5 0.5],'VerticalAlignment','bottom');
  end
  axis equal;
  axis tight;
  axis off;
  drawnow;
end
