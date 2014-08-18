function [hog_region_pyramid, im_region] = dwot_extract_region_fft(im, hog, scales,  bbsNMS, param)
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
  region_extraction_levels = 5;
end

% assume that bbsNMS is a matrix with row vectors
nBBox = size(bbsNMS, 1);
imSz = size(im);

% bbsNMS = round(clip_to_image(bbsNMS, [1 1 imSz(2) imSz(1)]));
% bbsNMS(:,1:4) = round(bbsNMS(:,1:4));
nHOG = numel(hog);

hog_region_pyramid = cell(1,nBBox);

for boxIdx = 1:nBBox

  % For the level that the detection occured,
  imgVIdx1 = bbsNMS(boxIdx, 1); % y1
  imgVIdx2 = bbsNMS(boxIdx, 3); % y1

  imgUIdx1 = bbsNMS(boxIdx, 2); % y1
  imgUIdx2 = bbsNMS(boxIdx, 4); % y1

  detScale = bbsNMS(boxIdx, 5); % scale
  detLevel = bbsNMS(boxIdx, 6); % level
  detUIdx  = bbsNMS(boxIdx, 7); % uus
  detVIdx  = bbsNMS(boxIdx, 8); % uus

  detTemplateIdx = bbsNMS(boxIdx, 11);
  detScore = bbsNMS(boxIdx, 12);
  
  startLevel = max(1, detLevel - region_extraction_levels);
  endLevel = min(nHOG, detLevel + region_extraction_levels);
  nLevel = endLevel - startLevel + 1;
  
  hog_region_pyramid{boxIdx}.imgBBox = bbsNMS(boxIdx,1:4);  
  hog_region_pyramid{boxIdx}.pyramid = struct('hogBBox',repmat({zeros(1,4)},nLevel,1),...
                                      'clipHogBBox',repmat({zeros(1,4)},nLevel,1),...
                                      'level',zeros(nLevel,1),...
                                      'scale',zeros(nLevel,1));
                           
  
  pyramidIdx = 1;
  
  % To extract pyramid, find the HOG index of image point (x1, y1) and (x2, y2)
  for level = startLevel:endLevel
    scale = scales(level);
    szHOG = size(hog{level});
    
    % U is row idx, V is col idx
    [hogUIdx1, hogVIdx1] = dwot_img_to_hog_fft(imgUIdx1, imgVIdx1, sbin, scale);
    [hogUIdx2, hogVIdx2] = dwot_img_to_hog_fft(imgUIdx2, imgVIdx2, sbin, scale);
    hogUIdx2 = hogUIdx2 - 1;
    hogVIdx2 = hogVIdx2 - 1;
    
    xpadding = ceil((hogVIdx2 - hogVIdx1) * region_extraction_padding_ratio);
    ypadding = ceil((hogUIdx2 - hogUIdx1) * region_extraction_padding_ratio);
    
    hogUIdx1 = floor(hogUIdx1);
    hogVIdx1 = floor(hogVIdx1); 
    hogUIdx2 = ceil(hogUIdx2);
    hogVIdx2 = ceil(hogVIdx2);
    
    clipU1 = max(hogUIdx1,1);
    clipV1 = max(hogVIdx1,1);
    clipU2 = min(hogUIdx2,szHOG(1));
    clipV2 = min(hogVIdx2,szHOG(2));
    
    % To save space, extract regions fromt the reference HOG.
    % hog_region_pyramid{pyramidIdx}.clipHog
    hog_region_pyramid{boxIdx}.pyramid(pyramidIdx).hogBBox = [hogVIdx1, hogUIdx1, hogVIdx2, hogUIdx2]; % x1 y1 x2 y2
    hog_region_pyramid{boxIdx}.pyramid(pyramidIdx).clipHogBBox = [clipV1, clipU1, clipV2, clipU2];
    hog_region_pyramid{boxIdx}.pyramid(pyramidIdx).paddedClipHogBBox = [clipV1 - xpadding, clipU1 - ypadding, clipV2 + xpadding, clipU2 - ypadding]; % x1 y1 x2 y2
    hog_region_pyramid{boxIdx}.pyramid(pyramidIdx).level = level;
    hog_region_pyramid{boxIdx}.pyramid(pyramidIdx).scale = scale;
    hog_region_pyramid{boxIdx}.pyramid(pyramidIdx).detTemplateIdx = detTemplateIdx;

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
      detectorIdx = bbsNMS(boxIdx, 11);
      imagesc(HOGpicture(param.detectors{detectorIdx}.whow));
      
      subplot(224);
      extractedHOG = hog{level}(round(hogUIdx1):round(hogUIdx2),...
                                round(hogVIdx1):round(hogVIdx2),:);
      imagesc(HOGpicture(extractedHOG,hogSize));
%       imagesc(HOGpicture(hog{level}(floor(detUIdx):ceil(detUIdx + templateSize{detId}(1) - 1),...
%                                     floor(detVIdx):ceil(detVIdx + templateSize{detId}(2) - 1),:), hogSize));
      
      a = fconvblasfloat( single(extractedHOG),  { single( param.detectors{detectorIdx}.whow)} , 1 ,1);
      fprintf('\ndetection level %d, cur level %d, \ndetection score %f, cur score %f\n',detLevel, level,  detScore, max(a{1}(:)));
        
%       if level == detLevel
%         fprintf('detection score single precision %f\n', bbsNMS(boxIdx, 12));
%         a = fconvblasfloat( single(extractedHOG),  { single( param.detectors{detectorIdx}.whow)} , 1 ,1);
%         fprintf('innerproduct score double precision %f\n', max(a{1}(:)));
%       end
      waitforbuttonpress;
    end    
  end
end

if nargout > 1 
  im_region = cell(1, nBBox);
  
  for boxIdx = 1:nBBox
    clip_box = clip_to_image( round(bbsNMS(boxIdx, 1:4)), [1 1 imSz(2) imSz(1)]);
    im_region{boxIdx} = im(clip_box(2):clip_box(4),clip_box(1):clip_box(3),:);
  end
end