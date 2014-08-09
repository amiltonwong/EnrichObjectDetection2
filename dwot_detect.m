function [bbsNMS, hog, scales] = dwot_detect(I, templates, param)

doubleIm = im2double(I);
[hog, scales] = esvm_pyramid(doubleIm, param);
padder = param.detect_pyramid_padding;
sbin = param.sbin;

nTemplates =  numel(templates);
sz = cellfun(@(x) size(x), templates, 'UniformOutput',false);

for level = 1:length(hog)
    hog{level} = padarray(hog{level}, [padder padder 0], 0); % Convolution, same size
end

minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), hog);
hog = hog(minsizes >= padder*2);
scales = scales(minsizes >= padder*2);
bbsAll = cell(length(hog),1);

for level = length(hog):-1:1
  HM = fconvblasfloat(single(hog{level}), templates, 1, nTemplates);

%      for modelIdx = 1:nTemplates
%        HM{modelIdx} = convnc(t.hog{level},flipTemplates{modelIdx},'valid');
%      end

  rmsizes = cellfun(@(x) size(x), HM, 'UniformOutput',false);
  scale = scales(level);
  templateIdxes = find(cellfun(@(x) prod(x), rmsizes));
  bbsTemplate = cell(nTemplates,1);
  
  for templateIdx = templateIdxes
    [idx] = find(HM{templateIdx}(:) > param.detection_threshold);
    if isempty(idx)
      continue;
    end

    [uus,vvs] = ind2sub(rmsizes{templateIdx}(1:2), idx);

    [y1, x1] = dwot_hog_to_img_conv(uus, vvs, sbin, scale, padder);
    [y2, x2] = dwot_hog_to_img_conv(uus + sz{templateIdx}(1), vvs + sz{templateIdx}(2), sbin, scale, padder);
    
    bbs = zeros(numel(uus), 12);
    bbs(:,1:4) = [x1 y1, x2, y2];
    
    bbs(:,5:12) = 0;
%     o = [uus vvs] - padder;
% 
%     bbs = ([o(:,2) o(:,1) o(:,2)+sz{templateIdx}(2) ...
%                o(:,1)+sz{templateIdx}(1)] - 1) * ...
%              sbin/scale + 1 + repmat([0 0 -1 -1],...
%               length(uus),1);

    bbs(:,5:12) = 0;

    bbs(:,5) = scale;
    bbs(:,6) = level;
    bbs(:,7) = uus;
    bbs(:,8) = vvs;

    % bbs(:,9) = boxoverlap(bbs, annotation.bbox + [0 0 annotation.bbox(1:2)]);
    % bbs(:,10) = abs(detectors{templateIdx}.az - azGT) < 30;

    bbs(:,11) = templateIdx;
    bbs(:,12) = HM{templateIdx}(idx);
    bbsTemplate{templateIdx} = bbs;
    
    % if visualize
    if 0
      [score, Idx] = max(bbs(:,12));
      % subplot(231); imagesc(templates{templateIdx}.rendering); axis equal; axis tight;
      % subplot(232); imagesc(detectors{exemplarIdx}.hogpic); axis equal; axis tight; axis off;
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

bbsNMS = esvm_nms(cell2mat(bbsAll),0.5);