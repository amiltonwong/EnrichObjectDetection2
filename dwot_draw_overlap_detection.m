function resultIm = dwot_draw_overlap_detection(im, bbsNMS, renderings, maxNDrawBox, drawPadding, box_text, drawing_weights)

if nargin < 4
  maxNDrawBox = 5;
end

if nargin  < 5
  drawPadding = 25;
end

if nargin < 6
  box_text = false;
end

if nargin < 7
  drawing_weights = ones(1,3)/3;
end

paddedIm = pad_image(im2double(im), drawPadding, 1);
resultIm = paddedIm;

NDrawBox = min(size(bbsNMS, 1),maxNDrawBox);

% Create overlap image
clipBBox = zeros(NDrawBox,4);
for bbsIdx = NDrawBox:-1:1
  if iscell(renderings)
    rendering = im2double(renderings{bbsNMS(bbsIdx, 11)});
  else
    rendering = im2double(renderings);
  end
  box_position = round(bbsNMS(bbsIdx, 1:4)) + drawPadding;
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
  cropRegion = round((box_position - clip_bnd) ./ [bboxWidth, bboxHeight, bboxWidth, bboxHeight] .* [renderingSz(2), renderingSz(1), renderingSz(2), renderingSz(1)]);
  resizeRendering = imresize(rendering(...
              (1 - cropRegion(2)):(end - cropRegion(4)),...
              (1 - cropRegion(1)):(end - cropRegion(3)),:),...
       [clip_bnd(4) - clip_bnd(2) + 1, clip_bnd(3) - clip_bnd(1) + 1]);
  resizeRendering(resizeRendering(:)>1)=1;
  resizeRendering(resizeRendering(:)<0)=0;
  
  % resizeRendering = resizeRendering(1:(clip_bnd(4) - clip_bnd(2) + 1), 1:(clip_bnd(3) - clip_bnd(1) + 1), :);
  bndIm = paddedIm( clip_bnd(2):clip_bnd(4), clip_bnd(1):clip_bnd(3), :);
  blendIm = bndIm * drawing_weights(1) +...
            im2double(resizeRendering) * drawing_weights(2)...
            + resultIm( clip_bnd(2):clip_bnd(4), clip_bnd(1):clip_bnd(3), :) * drawing_weights(3);
          
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
  color_map = hot(NDrawBox);
  for bbsIdx = NDrawBox:-1:1
    box_position = clipBBox(bbsIdx, 1:4) + [0 0 -clipBBox(bbsIdx, 1:2)];
    % if detector id id available print it
    if bbsNMS(bbsIdx,11) > 0
      box_text = sprintf(' s:%0.2f o:%0.2f t:%d',bbsNMS(bbsIdx,12),bbsNMS(bbsIdx,9),bbsNMS(bbsIdx,11));
    else
      box_text = sprintf(' s:%0.2f o:%0.2f ',bbsNMS(bbsIdx,12),bbsNMS(bbsIdx,9));
    end
    rectangle('position', box_position,'edgecolor',[0.5 0.5 0.5],'LineWidth',3);
    rectangle('position', box_position,'edgecolor',color_map(bbsIdx,:),'LineWidth',1);
    text(box_position(1) + 1 , box_position(2), box_text, 'BackgroundColor', color_map( bbsIdx ,:),'EdgeColor',[0.5 0.5 0.5],'VerticalAlignment','bottom');
  end
  axis equal;
  axis tight;
  axis off;
  drawnow;
end