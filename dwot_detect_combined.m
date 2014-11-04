function [bbsAllLevel, hog, scales] = dwot_detect_combined(I, templates_gpu, templates_cpu, param)

doubleIm = im2double(I);
[hog, scales] = esvm_pyramid(doubleIm, param);
padder = param.detect_pyramid_padding;
sbin = param.sbin;

if ~isfield(param, 'use_cpu_threshold')
  param.use_cpu_threshold = 10;
end

nTemplates =  numel(templates_gpu);

sz = cellfun(@(x) size(x), templates_gpu, 'UniformOutput',false);
maxTemplateHeight = max(cellfun(@(x) x(1), sz));
maxTemplateWidth = max(cellfun(@(x) x(2), sz));

minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), hog);
hog = hog(minsizes >= padder*2);
scales = scales(minsizes >= padder*2);
bbsAll = cell(length(hog),1);

for level = length(hog):-1:1
  % GPU
  if level > param.use_cpu_threshold
    fhog = cudaFFTData(single(hog{level}), maxTemplateHeight, maxTemplateWidth);
    HM = cudaConvFFTData(fhog,templates_gpu, [8, 8, 8, 16]);
  % CPU
  else
    hog{level} = padarray(single(hog{level}), [padder padder 0], 0); % Convolution, same size
    HM = fconvblasfloat(hog{level}, templates_cpu, 1, nTemplates);
  end

  rmsizes = cellfun(@(x) size(x), HM, 'UniformOutput',false);
  scale = scales(level);
  bbsTemplate = cell(nTemplates,1);
  
  for templateIdx = 1:nTemplates
    if param.b_calibrate
        HM{templateIdx} = dwot_calibrate_score(HM{templateIdx}, templateIdx, param.detectors, param);
    end
    
    [idx] = find(HM{templateIdx}(:) > param.detection_threshold);
    
    if isempty(idx)
      continue;
    end

    [uus,vvs] = ind2sub(rmsizes{templateIdx}(1:2), idx);

    % GPU
    if level > param.use_cpu_threshold
      [y1, x1] = dwot_hog_to_img_fft(uus - 1, vvs - 1, sz{templateIdx}, sbin, scale);
      [y2, x2] = dwot_hog_to_img_fft(uus + sz{templateIdx}(1) + 1, vvs + sz{templateIdx}(2) + 1, sz{templateIdx}, sbin, scale);
    % CPU
    else
      [y1, x1] = dwot_hog_to_img_conv(uus - 1, vvs - 1, sbin, scale, padder);
      [y2, x2] = dwot_hog_to_img_conv(uus + sz{templateIdx}(1) + 1, vvs + sz{templateIdx}(2) + 1, sbin, scale, padder);
    end

    bbs = zeros(numel(uus), 12);
    bbs(:,1:4) = [x1 y1, x2, y2];

    bbs(:,5) = scale;
    bbs(:,6) = level;
    bbs(:,7) = uus;
    bbs(:,8) = vvs;

    % bbs(:,10) is designated for GT index
    % bbs(:,9) = boxoverlap(bbs, annotation.bbox + [0 0 annotation.bbox(1:2)]);
    % bbs(:,10) = abs(detectors{templateIdx}.az - azGT) < 30;
    bbs(:,10) = param.detectors{templateIdx}.az;
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

bbsAllLevel = cell2mat(bbsAll);
