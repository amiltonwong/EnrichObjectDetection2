% Try new Renderer and ClassifierGenerator

% Initialize
if exist('renderer','var')
  renderer.delete();
  clear renderer;
end

addpath('3rdparty');
addpath('3rdparty/export_fig');
addpath('HoG');
addpath('HoG/features');
addpath('Renderer');

IMAGE_PATH = '../PBR_MATLAB/OBJECT_3D/';
IMAGE_START_IDX = 121;
IMAGE_END_IDX = 240;

% Setting
sbin = 6;
nLevel = 10;
visualize = false;
% visualize = true;
overlap = 0.5;
param = get_default_params(sbin,nLevel);
detectionThreshold = 100;

azs = 0:45:315;
azs = [azs azs - 10 azs + 10];
els = [0, 10, 20];
fovs = [15 30 60];
% yaw = 180;
yaw = 0;
if ismac
  yaw = 180;
end
nCellLimit = [150];
lambda = [0.02];

% Setup CG
load('sumGamma_N1_40_N2_40_sbin_4_nLevel_10_nImg_1601_napoli1_gamma.mat');
% load('sumGamma_N1_40_N2_40_sbin_4_nLevel_5_nImg_9963_napoli1_fft.mat');
load('mu.mat');
sbin = 6;

% detectorName = sprintf('detectors_%d_%d_fovs_honda_new_statistics.mat',nCellLimit, lambda);
detectorName = sprintf('detectors_%d_%d_fovs_honda.mat',nCellLimit, lambda);
if exist(detectorName,'file')
  load(detectorName);
  % templates  = cellfun(@(x) x.whow, detectors,'UniformOutput',false);
  templates  = cellfun(@(x) x.w, detectors,'UniformOutput',false);
else
  renderer = Renderer();
  renderer.initialize('Meshes/Honda-Accord-3.3ds', 700, 700, 0, 0, 0, 0, 25);
  detectors = {};
  templates = {};
  i = 1;
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

        templates{i} = WHOTemplate;  
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
  eval(sprintf('save detectors_%d_%d_fovs_honda_new_statistics.mat detectors',nCellLimit, lambda));
end

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

for i = 1:nTemplates
  bboxCollect{i} = [];
  nposv{i} = 0;
  tpv{i} = [];
  fpv{i} = [];
  apv{i} = [];
  afpv{i} = [];
  detScorev{i} = [];
  detIdxv{i} = [];
end


for imageIdx = IMAGE_START_IDX:IMAGE_END_IDX
  tic
  annotation = load([IMAGE_PATH '/Annotation/' sprintf('%04d',imageIdx) '.mat']);
  annotation = annotation.annotation;
  
  azGT = annotation.view(1);
  elGT = annotation.view(2);
  
  testDoubleIm = im2double(imread([IMAGE_PATH '/Images/' sprintf('%04d',imageIdx) '.jpg']));
  [t.hog, t.scales] = esvm_pyramid(testDoubleIm, param);
  t.padder = param.detect_pyramid_padding;
  for level = 1:length(t.hog)
      t.hog{level} = padarray(t.hog{level}, [t.padder t.padder 0], 0); % Convolution, same size
  end
  minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), t.hog);
  t.hog = t.hog(minsizes >= t.padder*2);
  t.scales = t.scales(minsizes >= t.padder*2);
  
  HM = {};
  bbsAll = [];
  for level = length(t.hog):-1:1
    HM = fconvblas(t.hog{level}, templates, 1, nTemplates);
%      for modelIdx = 1:nTemplates
%        HM{modelIdx} = convnc(t.hog{level},flipTemplates{modelIdx},'valid');
%      end
    
    rmsizes = cellfun(@(x) size(x), HM, 'UniformOutput',false);
    scale = t.scales(level);
    templateIdxes = find(cellfun(@(x) prod(x), rmsizes));
    
    for exemplarIdx = templateIdxes
      if numel(HM{exemplarIdx}) == 0
        continue;
      end
      [idx] = find(HM{exemplarIdx}(:) > detectionThreshold);
      
      if isempty(idx)
        continue;
      end
      
      [uus,vvs] = ind2sub(rmsizes{exemplarIdx}(1:2), idx);

      o = [uus vvs] - t.padder;

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
      bbsAll = [bbsAll; bbs];
      bboxCollect{exemplarIdx} = [bboxCollect{exemplarIdx} ; bbs];
      
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
  bbsNMS = esvm_nms(bbsAll,0.5);
  
  % number of ground truth boxes
  npos = npos + 1;

  % number of detection after NMS
  detected = false;
  nDet = size(bbsNMS,1);
  cellIdx = imageIdx - IMAGE_START_IDX + 1;
  tp{cellIdx} = zeros(1,nDet);
  fp{cellIdx} = zeros(1,nDet);
  atp{cellIdx} = zeros(1,nDet);
  afp{cellIdx} = zeros(1,nDet);
  detectorId{cellIdx} = bbsNMS(:,11)';
  detScore{cellIdx} = bbsNMS(:,end)';
  
  for bbsIdx = 1:nDet
    if bbsNMS(bbsIdx,9) > 0.5 && ~detected
      tp{cellIdx}(bbsIdx) = 1;
      if bbsNMS(bbsIdx,10) == 1
        detected = true;
        atp{cellIdx}(bbsIdx) = 1;
      else
        afp{cellIdx}(bbsIdx) = 1;
      end
    else
      fp{cellIdx}(bbsIdx) = 1;
      afp{cellIdx}(bbsIdx) = 1;
    end
  end
  
  if visualize
    for bbsIdx = size(bbsNMS,1):-1:1
      clf;
      exemplarIdx = bbsNMS(bbsIdx,11);
      subplot(121); imagesc(detectors{exemplarIdx}.rendering); axis equal; axis tight;

      % subplot(222); imagesc(detectors{exemplarIdx}.hogpic); axis equal; axis tight; axis off;

      text(10,20,{['score ' num2str( bbsNMS(bbsIdx,13))], ...
        ['overlap ' num2str( bbsNMS(bbsIdx,9))],...
        ['azimuth D/GT ' num2str( detectors{exemplarIdx}.az) ' ' num2str(azGT)],...
        ['azimuth ' num2str(bbsNMS(bbsIdx,10))]},...
        'BackgroundColor',[.7 .9 .7]);
      
      subplot(122); imagesc(testDoubleIm); axis equal; axis tight; axis off;
      bbs = bbsNMS(bbsIdx,:);
      rectangle('Position',bbs(1,1:4)-[0 0 bbs(1,1:2)]);
      pause(0.5);
      % waitforbuttonpress;
    end
  end

  fprintf('%d \n',imageIdx);
  toc
end

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
