function [bbsAllLevel, hog, scales] = dwot_detect(I, templates, param)

doubleIm = im2double(I);
[hog, scales] = esvm_pyramid(doubleIm, param);
hogPadder = param.detect_pyramid_padding;
sbin = param.sbin;

nTemplates =  numel(templates);
sz = cellfun(@(x) size(x), templates, 'UniformOutput',false);

minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), hog);
hog = hog(minsizes >= param.min_hog_length);
scales = scales(minsizes >= param.min_hog_length);
bbsAll = cell(length(hog),1);

for level = length(hog):-1:1
  hog{level} = padarray(single(hog{level}), [hogPadder hogPadder 0], 0); % Convolution, same size
  HM = fconvblasfloat(hog{level}, templates, 1, nTemplates);

%      for modelIdx = 1:nTemplates
%        HM{modelIdx} = convnc(t.hog{level},flipTemplates{modelIdx},'valid');
%      end

  rmsizes = cellfun(@(x) size(x), HM, 'UniformOutput',false);
  scale = scales(level);
  templateIdxes = find(cellfun(@(x) prod(x), rmsizes));
  bbsTemplate = cell(nTemplates,1);
  
  for templateIdx = templateIdxes
    % Use calibration
    if param.b_calibrate
        HM{templateIdx} = dwot_calibrate_score(HM{templateIdx}, templateIdx, param.detectors, param);
    end
    
    [idx] = find(HM{templateIdx}(:) > param.detection_threshold);
    if isempty(idx)
      continue;
    end

    [y_coord,x_coord] = ind2sub(rmsizes{templateIdx}(1:2), idx);

    % HOG templates are consistently smaller. Add extra padding after
    % detection
    [y1, x1] = dwot_hog_to_img_conv(y_coord - 1, x_coord -1, sbin, scale, hogPadder);
    [y2, x2] = dwot_hog_to_img_conv(y_coord + sz{templateIdx}(1) + 1, x_coord + sz{templateIdx}(2) + 1, sbin, scale, hogPadder);
    
    bbs = zeros(numel(y_coord), 12);
    bbs(:,1:4) = [x1 y1, x2, y2];
    
%     o = [uus vvs] - hogPadder;
% 
%     bbs = ([o(:,2) o(:,1) o(:,2)+sz{templateIdx}(2) ...
%                o(:,1)+sz{templateIdx}(1)] - 1) * ...
%              sbin/scale + 1 + repmat([0 0 -1 -1],...
%               length(uus),1);
%     bbs(:,5:12) = 0;

    bbs(:,5) = scale;
    bbs(:,6) = level;
    bbs(:,7) = y_coord;
    bbs(:,8) = x_coord;

    % bbs(:,9) is designated for overlap
    % bbs(:,10) is designated for GT index
    
    % bbs(:,9) = boxoverlap(bbs, annotation.bbox + [0 0 annotation.bbox(1:2)]);
    % bbs(:,10) = abs(detectors{templateIdx}.az - azGT) < 30;

    bbs(:,11) = templateIdx;
    bbs(:,12) = HM{templateIdx}(idx);

    bbsTemplate{templateIdx} = bbs;
    
    % if visualize
    if 0
      subplot(231); imagesc(HOGpicture(templates{templateIdx})); axis equal; axis tight;
      subplot(232); imagesc(param.detectors{templateIdx}.rendering_image); axis equal; axis tight; axis off;
      text(10,20,{['score ' num2str(bbs(Idx,12))],['azimuth ' num2str(bbs(Idx,10))]},'BackgroundColor',[.7 .9 .7]);
      subplot(233); imagesc(HM{templateIdx}); %caxis([100 200]); 
      colorbar; axis equal; axis tight; 
      subplot(234); imagesc(doubleIm); axis equal; % axis tight; axis off;
      rectangle('Position',bbs(Idx,1:4)-[0 0 bbs(Idx,1:2)]);
      % subplot(235); imagesc(HOGpic); axis equal; axis tight; axis off;
      % waitforbuttonpress;
      pause(0.8);
    end
  end
  bbsAll{level} = cell2mat(bbsTemplate);
end

bbsAllLevel = cell2mat(bbsAll);
