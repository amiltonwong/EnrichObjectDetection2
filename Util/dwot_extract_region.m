% Deprecated
% use dwot_extract_hog
function [hog_region_pyramid, im_region]= dwot_extract_region(im, hog, scales, bbs_nms, param)
% Clip bounding box to fit image.

% Create HOG pyramid for each of the proposal regions.
% hog_region : 
% im_region : 

% Padded Region
%  -------------------
%  | offset x, y
%  | 
%  |  ----- Actual image and hog region start
%  |  |
% To prevent unnecessary 


padder = param.detect_pyramid_padding;
sbin = param.sbin;

if isfield(param,'region_extraction_padding_ratio')
  region_extraction_padding_ratio = param.extraction_padding_ratio;
else
  region_extraction_padding_ratio = 0.1; % 10 percent
end

if isfield(param, 'region_extraction_levels')
  region_extraction_levels = param.region_extraction_levels;
else
  region_extraction_levels = 2;
end

% assume that bbs_nms is a matrix with row vectors
nBBox = size(bbs_nms, 1);
imSz = size(im);

% bbs_nms = round(clip_to_image(bbs_nms, [1 1 imSz(2) imSz(1)]));
% bbs_nms(:,1:4) = round(bbs_nms(:,1:4));
nHOG = numel(hog);

hog_region_pyramid = cell(1,nBBox);

for boxIdx = 1:nBBox

  % For the level that the detection occured,
  imgVIdx1 = bbs_nms(boxIdx, 1); % y1
  imgVIdx2 = bbs_nms(boxIdx, 3); % y1

  imgUIdx1 = bbs_nms(boxIdx, 2); % y1
  imgUIdx2 = bbs_nms(boxIdx, 4); % y1

  detScale = bbs_nms(boxIdx, 5); % scale
  detLevel = bbs_nms(boxIdx, 6); % level
  hog_x_coord  = bbs_nms(boxIdx, 7); % y-coord
  hog_y_coord  = bbs_nms(boxIdx, 8); % x-coord
  detTemplateIdx = bbs_nms(boxIdx, 11); % template Id
  detScore = bbs_nms(boxIdx, 12);
  
  startLevel = max(1, detLevel - region_extraction_levels);
  endLevel = min(nHOG, detLevel + region_extraction_levels);
  nLevel = endLevel - startLevel + 1;
  
  hog_region_pyramid{boxIdx}.detection_level = detLevel;
  hog_region_pyramid{boxIdx}.detection_level_hog_coord_x = hog_x_coord;
  hog_region_pyramid{boxIdx}.detection_level_hog_coord_y = hog_y_coord;
  hog_region_pyramid{boxIdx}.imgBBox = bbs_nms(boxIdx,1:4);  
  hog_region_pyramid{boxIdx}.pyramid = struct('hog_bbox',repmat({zeros(1,4)},nLevel,1),...
                                      'clip_hog_bbox',repmat({zeros(1,4)},nLevel,1),...
                                      'level',zeros(nLevel,1),...
                                      'scale',zeros(nLevel,1));
                           
  
  pyramidIdx = 1;
  
  % To extract pyramid, find the HOG index of image point (x1, y1) and (x2, y2)
  for level = startLevel:endLevel
    scale = scales(level);
    szHOG = size(hog{level});
    
    % U is row idx, V is col idx
    if param.computing_mode == 0
      [hogUIdx1, hogVIdx1] = dwot_img_to_hog_conv(imgUIdx1, imgVIdx1, sbin, scale, padder);
      [hogUIdx2, hogVIdx2] = dwot_img_to_hog_conv(imgUIdx2, imgVIdx2, sbin, scale, padder);
    elseif param.computing_mode == 1
      [hogUIdx1, hogVIdx1] = dwot_img_to_hog_fft(imgUIdx1, imgVIdx1, sbin, scale);
      [hogUIdx2, hogVIdx2] = dwot_img_to_hog_fft(imgUIdx2, imgVIdx2, sbin, scale);
    else
      error('computing mode not defined');
    end
    
    hogUIdx2 = hogUIdx2 - 1;
    hogVIdx2 = hogVIdx2 - 1;
    
    xpadding = ceil((hogVIdx2 - hogVIdx1) * region_extraction_padding_ratio);
    ypadding = ceil((hogUIdx2 - hogUIdx1) * region_extraction_padding_ratio);
    
    clipU1 = max(floor(hogUIdx1),1);
    clipV1 = max(floor(hogVIdx1),1);
    clipU2 = min(ceil(hogUIdx2),szHOG(1));
    clipV2 = min(ceil(hogVIdx2),szHOG(2));
    
    % To save space, extract regions fromt the reference HOG.
    % hog_region_pyramid{pyramidIdx}.clipHog
    hog_region_pyramid{boxIdx}.pyramid(pyramidIdx).hog_bbox = [hogVIdx1, hogUIdx1, hogVIdx2, hogUIdx2]; % x1 y1 x2 y2
    hog_region_pyramid{boxIdx}.pyramid(pyramidIdx).clip_Hog_bbox = [clipV1, clipU1, clipV2, clipU2];
    hog_region_pyramid{boxIdx}.pyramid(pyramidIdx).padded_clip_hog_bbox = [clipV1 - xpadding, clipU1 - ypadding, clipV2 + xpadding, clipU2 - ypadding]; % x1 y1 x2 y2
    hog_region_pyramid{boxIdx}.pyramid(pyramidIdx).level = level;
    hog_region_pyramid{boxIdx}.pyramid(pyramidIdx).scale = scale;
    hog_region_pyramid{boxIdx}.pyramid(pyramidIdx).template_idx = detTemplateIdx;
    
    pyramidIdx = pyramidIdx + 1;
    % debug
    if 1
      figure(1);
      subplot(221);
      imagesc(im);
      rectangle('position',[imgVIdx1, imgUIdx1, imgVIdx2-imgVIdx1, imgUIdx2-imgUIdx1]);
      
      subplot(222);
      hogSize = 20;
      imagesc(HOGpicture(hog{level},hogSize));
      dwot_draw_hog_bounding_box(hogVIdx1, hogUIdx1, hogVIdx2, hogUIdx2, hogSize);
      title(['level : ' num2str(level) ' detlevel : ' num2str(detLevel)]);
      
      subplot(223);
      detectorIdx = bbs_nms(boxIdx, 11);
      imagesc(HOGpicture(param.detectors{detectorIdx}.whow));
      
      subplot(224);
      extractedHOG = hog{level}(floor(hogUIdx1):ceil(hogUIdx2), floor(hogVIdx1):ceil(hogVIdx2),:);
      imagesc(HOGpicture(extractedHOG,hogSize));
      
      if level == detLevel
        fprintf('detection score single precision %f\n', bbs_nms(boxIdx, 12));
        fprintf('innerproduct score double precision %f\n', param.detectors{detectorIdx}.whow(:)' *  extractedHOG(:));
      end
      waitforbuttonpress;
    end    
  end
end

if nargout > 1 
  im_region = cell(1, nBBox);
  
  for boxIdx = 1:nBBox
    clip_box = clip_to_image( round(bbs_nms(boxIdx, 1:4)), [1 1 imSz(2) imSz(1)]);
    im_region{boxIdx} = im(clip_box(2):clip_box(4),clip_box(1):clip_box(3),:);
  end
end