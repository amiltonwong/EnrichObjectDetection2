function [hog_region, im_region]= dwot_extract_region(im, bbsNMS, param)
% Create HOG pyramid for each of the proposal regions.
% hog_region : 
% im_region : 

% Padded Region
%  ┌─────────────────────
%  | offset x, y
%  | ↘
%  |   ┌─── Actual image and hog region start
%  |   |
% To prevent unnecessary 

if isfield(param,'extraction_padding')
  extraction_padding = param.extraction_padding;
else
  extraction_padding = 0.1; % 10 percent
end

% assume that bbsNMS is a matrix with row vectors
nBBox = size(bbsNMS, 1);
imSz = size(im);

bbsNMS_clip = round(clip_to_image(bbsNMS, [1 1 imSz(2) imSz(1)]));

for boxIdx = 1:nBBox
  
end

if nargout > 1 
  im_region = cell(1, nBBox);
  
  for boxIdx = 1:nBBox
    im_region{boxIdx} = bbsNMS_clip(boxIdx, 1:4)
  end
end