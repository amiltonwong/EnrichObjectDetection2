function [top, pick] = dwot_per_view_nms(boxes, overlap)
% top = esvm_nms(boxes, overlap)
% Non-maximum suppression. (FAST VERSION)
% Greedily select high-scoring detections and skip detections
% that are significantly covered by a previously selected
% detection.
% NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
% but an inner loop has been eliminated to significantly speed it
% up in the case of a large number of boxes

% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm


if isempty(boxes)
  top = [];
  return;
end

unique_templates = unique(boxes(:,11));
n_unique = numel(unique_templates);
top = cell(n_unique,1);

for unique_template_index = 1:n_unique
  unique_template = unique_templates(unique_template_index);
  box_indexes = (unique_template == boxes(:,11));
  boxes_per_template = boxes(box_indexes,:);
  x1 = boxes_per_template(:,1);
  y1 = boxes_per_template(:,2);
  x2 = boxes_per_template(:,3);
  y2 = boxes_per_template(:,4);
  s = boxes_per_template(:,end);

  area = (x2-x1+1) .* (y2-y1+1);
  [vals, I] = sort(s);

  pick = s*0;
  counter = 1;
  while ~isempty(I)

    last = length(I);
    i = I(last);  
    pick(counter) = i;
    counter = counter + 1;

    xx1 = max(x1(i), x1(I(1:last-1)));
    yy1 = max(y1(i), y1(I(1:last-1)));
    xx2 = min(x2(i), x2(I(1:last-1)));
    yy2 = min(y2(i), y2(I(1:last-1)));

    w = max(0.0, xx2-xx1+1);
    h = max(0.0, yy2-yy1+1);

    o = w.*h ./ area(I(1:last-1));

    I([last; find(o>overlap)]) = [];
  end

  pick = pick(1:(counter-1));
  top{unique_template_index} = boxes_per_template(pick,:);
end