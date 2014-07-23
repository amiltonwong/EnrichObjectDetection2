addpath('HoG');
addpath('HoG/features');
addpath('Util');
addpath('../MatlabRenderer/');
% addpath('/home/chrischoy/Dataset/VOCdevkit/');
% IMAGE_PATH = '/home/chrischoy/Dataset/VOCdevkit/VOC2007/';
IMAGE_PATH = '../PBR_MATLAB/OBJECT_3D/';

load('Statistics/sumGamma_N1_40_N2_40_sbin_4_nLevel_10_nImg_1601_napoli1_gamma.mat');
% load('sumGamma_N1_40_N2_40_sbin_4_nLevel_5_nImg_9963_napoli1_fft.mat');

% Setting
sbin = 6;
nLevel = 10;
visualize = false;
% visualize = true;
overlap = 0.5;
param = get_default_params(sbin,nLevel);
detectionThreshold = 100;

azs = 0:45:315;
azs = [azs , azs - 10, azs + 10];
els = [0 10 20];
fovs = [15 30 60];
% yaw = 180;
yaw = 0;
if ismac
  yaw = 180;
end
nCellLimit = [150];
lambda = [0.02];
IMAGE_START_IDX = 121;
IMAGE_END_IDX = 240;

% detectorName = sprintf('detectors_%d_%d_fovs_honda_new_statistics.mat',nCellLimit, lambda);
detectorName = sprintf('detectors_%d_%0.4f_fovs_a_%d_e_%d_f_%d_honda.mat',nCellLimit, lambda, numel(azs), numel(els), numel(fovs));
if exist(detectorName,'file')
  load(detectorName);
else
  if exist('renderer','var')
    renderer.delete();
    clear renderer;
  end
  
  renderer = Renderer();
  if ~renderer.initialize('Mesh/Car/Sedan/Honda-Accord-3.3ds', 700, 700, 0, 0, 0, 0, 25)
    error('fail to load model');
  end
  
  i = 1;
  detectors = cell(1,numel(azs) * numel(els) * numel(fovs));
  for azIdx = 1:numel(azs)
    for elIdx = 1:numel(els)
      for fovIdx = 1:numel(fovs)
        elGT = els(elIdx);
        azGT = azs(azIdx);
        fovGT = fovs(fovIdx);
        tic
        renderer.setViewpoint(90-azGT,elGT,yaw,0,fovGT);
        im = renderer.renderCrop();
        [ WHOTemplate, HOGTemplate] = WHOTemplateDecomp( im, mu, Gamma, nCellLimit, lambda, 50);
        toc;

        detectors{i}.whow = WHOTemplate;
        % detectors{i}.hogw = HOGTemplate;
        detectors{i}.az = azGT;
        detectors{i}.el = elGT;
        detectors{i}.fov = fovGT;
        detectors{i}.rendering = im;
        detectors{i}.sz = size(WHOTemplate);
        % detectors{i}.whopic = HOGpicture(WHOTemplate);

        if visualize
          figure(1); subplot(131);
          imagesc(im); axis equal; axis tight;
          subplot(132);
          imagesc(HOGpicture(HOGTemplate)); axis equal; axis tight;
          subplot(133);
          imagesc(HOGpicture(WHOTemplate)); axis equal; axis tight;
          waitforbuttonpress;
        end
        i = i + 1;    
      end
    end
  end
  eval(sprintf(['save ' detectorName ' detectors'],nCellLimit, lambda));
end

templates  = cellfun(@(x) x.whow, detectors,'UniformOutput',false);
% templates  = cellfun(@(x) x.w, detectors,'UniformOutput',false);

nTemplates =  numel(templates);
sz = cellfun(@(x) size(x), templates, 'UniformOutput',false);
bboxCollect = {};

N_IMAGE = IMAGE_END_IDX - IMAGE_START_IDX + 1;

npos = 0;
tp = cell(1,N_IMAGE);
fp = cell(1,N_IMAGE);
atp = cell(1,N_IMAGE);
afp = cell(1,N_IMAGE);
detScore = cell(1,N_IMAGE);
detectorId = cell(1,N_IMAGE);
detIdx = 0;

% for i = 1:nTemplates
%   bboxCollect{i} = [];
%   nposv{i} = 0;
%   tpv{i} = [];
%   fpv{i} = [];
%   apv{i} = [];
%   afpv{i} = [];
%   detScorev{i} = [];
%   detIdxv{i} = [];
% end

startTime = clock;
for imageIdx = 1:(IMAGE_END_IDX - IMAGE_START_IDX + 1)
  tic
  annotation = load([IMAGE_PATH '/Annotation/' sprintf('%04d',imageIdx + IMAGE_START_IDX - 1) '.mat']);
  annotation = annotation.annotation;
  
  azGT = annotation.view(1);
  elGT = annotation.view(2);
  
  testDoubleIm = im2double(imread([IMAGE_PATH '/Images/' sprintf('%04d',imageIdx + IMAGE_START_IDX - 1) '.jpg']));
  [hog, scales] = esvm_pyramid(testDoubleIm, param);
  padder = param.detect_pyramid_padding;
  for level = 1:length(scales)
      hog{level} = padarray(hog{level}, [padder padder 0], 0); % Convolution, same size
  end
  minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), hog);
  hog = hog(minsizes >= padder*2);
  scales = scales(minsizes >= padder*2);
  
  HM = {};
  bbsAll = cell(length(hog),1);
  for level = length(hog):-1:1
    HM = fconvblas(hog{level}, templates, 1, nTemplates);
%      for modelIdx = 1:nTemplates
%        HM{modelIdx} = convnc(t.hog{level},flipTemplates{modelIdx},'valid');
%      end
    
    rmsizes = cellfun(@(x) size(x), HM, 'UniformOutput', false);
    scale = scales(level);
    templateIdxes = find(cellfun(@(x) prod(x), rmsizes));
    
    for exemplarIdx = templateIdxes
      [idx] = find(HM{exemplarIdx}(:) > detectionThreshold);
      if isempty(idx)
        continue;
      end
     
      [uus,vvs] = ind2sub(rmsizes{exemplarIdx}(1:2), idx);

      o = [uus vvs] - padder;

      bbs = ([o(:,2) o(:,1) o(:,2)+sz{exemplarIdx}(2) ...
                 o(:,1)+sz{exemplarIdx}(1)] - 1) * ...
               sbin/scale + 1 + repmat([0 0 -1 -1],...
                length(uus),1);
      
      bbs(:,5:13) = 0;
      
      bbs(:,5) = scale;
      bbs(:,6) = level;
      bbs(:,7) = uus;
      bbs(:,8) = vvs;
      
      bbs(:,9) = boxoverlap(bbs, annotation.bbox + [0 0 annotation.bbox(1:2)]);
      bbs(:,10) = abs(detectors{exemplarIdx}.az - azGT) < 30;
      bbs(:,11) = exemplarIdx;
      bbs(:,12) = imageIdx;
      bbs(:,13) = HM{exemplarIdx}(idx);
      bbsAll{level} = bbs;
      % bboxCollect{exemplarIdx} = [bboxCollect{exemplarIdx} ; bbs];
      
       % if visualize
       if 0
        [score, Idx] = max(bbs(:,13));
        subplot(231); imagesc(detectors{exemplarIdx}.rendering); axis equal; axis tight;
        % subplot(232); imagesc(detectors{exemplarIdx}.hogpic); axis equal; axis tight; axis off;
        text(10,20,{['score ' num2str(bbs(Idx,13))], ['overlap ' num2str(bbs(Idx,9))],['azimuth ' num2str(bbs(Idx,10))]},'BackgroundColor',[.7 .9 .7]);
        subplot(233); imagesc(HM{exemplarIdx}); %caxis([100 200]); 
        colorbar; axis equal; axis tight; 
        subplot(234); imagesc(testDoubleIm); axis equal; % axis tight; axis off;
        rectangle('Position',bbs(Idx,1:4)-[0 0 bbs(Idx,1:2)]);
        % subplot(235); imagesc(HOGpic); axis equal; axis tight; axis off;
        % waitforbuttonpress;
        % pause(0.8);
      end
    end
  end
  bbsAll = cell2mat(bbsAll);
  bbsNMS = esvm_nms(bbsAll,0.5);
  
  % number of ground truth boxes
  npos = npos + 1;

  % number of detection after NMS
  detected = false;
  nDet = size(bbsNMS,1);
  tp{imageIdx} = zeros(1,nDet);
  fp{imageIdx} = zeros(1,nDet);
  atp{imageIdx} = zeros(1,nDet);
  afp{imageIdx} = zeros(1,nDet);
  detectorId{imageIdx} = bbsNMS(:,11)';
  detScore{imageIdx} = bbsNMS(:,end)';
  
  for bbsIdx = 1:nDet
    if bbsNMS(bbsIdx,9) > 0.5 && ~detected
      tp{imageIdx}(bbsIdx) = 1;
      if bbsNMS(bbsIdx,10) == 1
        detected = true;
        atp{imageIdx}(bbsIdx) = 1;
      else
        afp{imageIdx}(bbsIdx) = 1;
      end
    else
      fp{imageIdx}(bbsIdx) = 1;
      afp{imageIdx}(bbsIdx) = 1;
    end
  end
  
  if visualize
  % if 1
    padding = 100;
    paddedIm = pad_image(testDoubleIm, padding, 1);
    resultIm = paddedIm;
    NDrawBox = min(nDet,2);
    for bbsIdx = NDrawBox:-1:1
      % rectangle('position', bbsNMS(bbsIdx, 1:4) - [0 0 bbsNMS(bbsIdx, 1:2)] + [padding padding 0 0]);
      bnd = round(bbsNMS(bbsIdx, 1:4)) + padding;
      szIm = size(paddedIm);
      clip_bnd = [ min(bnd(1),szIm(2)),...
          min(bnd(2), szIm(1)),...
          min(bnd(3), szIm(2)),...
          min(bnd(4), szIm(1))];
      clip_bnd = [max(clip_bnd(1),1),...
          max(clip_bnd(2),1),...
          max(clip_bnd(3),1),...
          max(clip_bnd(4),1)];
      % resizeRendering = imresize(detectors{bbsNMS(bbsIdx, 11)}.rendering, [bnd(4) - bnd(2) + 1, bnd(3) - bnd(1) + 1]);
      resizeRendering = imresize(detectors{bbsNMS(bbsIdx, 11)}.r, [bnd(4) - bnd(2) + 1, bnd(3) - bnd(1) + 1]);
      resizeRendering = resizeRendering(1:(clip_bnd(4) - clip_bnd(2) + 1), 1:(clip_bnd(3) - clip_bnd(1) + 1), :);
      bndIm = paddedIm( clip_bnd(2):clip_bnd(4), clip_bnd(1):clip_bnd(3), :);
      blendIm = bndIm/2 + im2double(resizeRendering)/2;
      resultIm(clip_bnd(2):clip_bnd(4), clip_bnd(1):clip_bnd(3),:) = blendIm;
    end
    clf;
    imagesc(resultIm);
    
    for bbsIdx = NDrawBox:-1:1
      bnd = round(bbsNMS(bbsIdx, 1:4)) + padding;
      szIm = size(paddedIm);
      clip_bnd = [ min(bnd(1),szIm(2)),...
          min(bnd(2), szIm(1)),...
          min(bnd(3), szIm(2)),...
          min(bnd(4), szIm(1))];
      clip_bnd = [max(clip_bnd(1),1),...
          max(clip_bnd(2),1),...
          max(clip_bnd(3),1),...
          max(clip_bnd(4),1)];
      titler = {['score ' num2str( bbsNMS(bbsIdx,13))], ...
        [' overlap ' num2str( bbsNMS(bbsIdx,9))],...
        [' azimuth D/GT ' num2str( detectors{exemplarIdx}.az) ' ' num2str(azGT)],...
        [' azimuth ' num2str(bbsNMS(bbsIdx,10))]};
      
      plot_bbox(clip_bnd,cell2mat(titler),[1 1 1]);
    end
    drawnow;
    disp('Press any button to continue');
    waitforbuttonpress;
    
%     for bbsIdx = size(bbsNMS,1):-1:1
%       clf;
%       exemplarIdx = bbsNMS(bbsIdx,11);
%       subplot(121); imagesc(detectors{exemplarIdx}.rendering); axis equal; axis tight;
% 
%       % subplot(222); imagesc(detectors{exemplarIdx}.hogpic); axis equal; axis tight; axis off;
% 
%       text(10,20,{['score ' num2str( bbsNMS(bbsIdx,13))], ...
%         ['overlap ' num2str( bbsNMS(bbsIdx,9))],...
%         ['azimuth D/GT ' num2str( detectors{exemplarIdx}.az) ' ' num2str(azGT)],...
%         ['azimuth ' num2str(bbsNMS(bbsIdx,10))]},...
%         'BackgroundColor',[.7 .9 .7]);
%       
%       subplot(122); imagesc(testDoubleIm); axis equal; axis tight; axis off;
%       bbs = bbsNMS(bbsIdx,:);
%       rectangle('Position',bbs(1,1:4)-[0 0 bbs(1,1:2)]);
%       pause(0.5);
%       % waitforbuttonpress;
%     end
  end

  fprintf('%d ',imageIdx);
  toc
end
endTime = clock;
etime(endTime, startTime)

detScore = cell2mat(detScore);
fp = cell2mat(fp);
tp = cell2mat(tp);
atp = cell2mat(atp);
afp = cell2mat(afp);
detectorId = cell2mat(detectorId);

[sc, si] =sort(detScore,'descend');
fpSort = cumsum(fp(si));
tpSort = cumsum(tp(si));

atpSort = cumsum(atp(si));
afpSort = cumsum(afp(si));

detectorIdSort = detectorId(si);

recall = tpSort/npos;
precision = tpSort./(fpSort + tpSort);

arecall = atpSort/npos;
aprecision = atpSort./(afpSort + atpSort);

ap = VOCap(recall', precision');
aa = VOCap(arecall', aprecision');
fprintf('AP = %.4f\n', ap);
plot(recall, precision, 'r', 'LineWidth',3);
hold on;
plot(arecall, aprecision, 'g', 'LineWidth',3);
xlabel('Recall');
ylabel('Precision/Accuracy');
tit = sprintf('Average Precision = %.1f / Average Accuracy = %1.1f', 100*ap,100*aa);
title(tit);
axis([0 1 0 1]);
set(gcf,'color','w');
saveas(gcf,sprintf('Result/valSet_NCells_%d_lambda_%0.4f_azs_%d_els_%d_fovs_%d_honda_accord.png',...
  nCellLimit,lambda,numel(azs),numel(els),numel(fovs)));