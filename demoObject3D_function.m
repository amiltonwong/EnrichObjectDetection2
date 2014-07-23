% VOC_PATH = '/home/chrischoy/Dataset/VOCdevkit/';
clear;
addpath('HoG');
addpath('HoG/features');
addpath('Util');
addpath('../MatlabRenderer/');

IMAGE_PATH = '../PBR_MATLAB/OBJECT_3D/';
IMAGE_START_IDX = 121;
IMAGE_END_IDX = 240;
N_IMAGE = IMAGE_END_IDX - IMAGE_START_IDX + 1;

azs = 0:45:315; % azs = [azs , azs - 10, azs + 10];
els = [0 10 20];
fovs = [15 30 60];
yaws = ismac * 180;
n_cell_limit = [150];
lambda = [0.02];
visualize = true;
% visualize = false;
sbin = 6;
nlevel = 10;
detection_threshold = 100;

model_file = 'Mesh/Car/Sedan/Honda-Accord-3';
model_name = strrep(model_file, '/', '_');

detector_name = sprintf('%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d.mat',...
    model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs));

if exist(detector_name,'file')
  load(detector_name);
else
  load('Statistics/sumGamma_N1_40_N2_40_sbin_4_nLevel_10_nImg_1601_napoli1_gamma.mat');
  detectors = dwot_make_detectors_slow(mu, Gamma, [model_file '.3ds'], azs, els, yaws, fovs, n_cell_limit, lambda, visualize);
  if sum(cellfun(@(x) isempty(x), detectors))
    error('Detector Not Completed');
  end
  eval(sprintf(['save ' detector_name ' detectors']));
end

templates = cellfun(@(x) x.whow, detectors,'UniformOutput',false);
param = get_default_params(sbin, nlevel, detection_threshold);

% extract ground truth objects
npos = 0;
tp = cell(1,N_IMAGE);
fp = cell(1,N_IMAGE);
atp = cell(1,N_IMAGE);
afp = cell(1,N_IMAGE);
detScore = cell(1,N_IMAGE);
detectorId = cell(1,N_IMAGE);
detIdx = 0;


startTime = clock;
for imgIdx = 1:(IMAGE_END_IDX - IMAGE_START_IDX + 1)
    imgTic = tic;
    % read annotation
    annotation = load([IMAGE_PATH '/Annotation/' sprintf('%04d',imgIdx + IMAGE_START_IDX - 1) '.mat']);
    annotation = annotation.annotation;

    azGT = annotation.view(1);
    elGT = annotation.view(2);
    
    im = imread([IMAGE_PATH '/Images/' sprintf('%04d',imgIdx + IMAGE_START_IDX - 1) '.jpg']);
    [bbsNMS ]= dwot_detect( im, templates, param);
    
    npos=npos+1;

    nDet = size(bbsNMS,1);
    tp{imgIdx} = zeros(1,nDet);
    fp{imgIdx} = zeros(1,nDet);
%     atp{imgIdx} = zeros(1,nDet);
%     afp{imgIdx} = zeros(1,nDet);
    detectorId{imgIdx} = bbsNMS(:,11)';
    detScore{imgIdx} = bbsNMS(:,end)';
    
    detected = false;
    nDet = size(bbsNMS,1);
    tp{imgIdx} = zeros(1,nDet);
    fp{imgIdx} = zeros(1,nDet);
    atp{imgIdx} = zeros(1,nDet);
    afp{imgIdx} = zeros(1,nDet);
    detectorId{imgIdx} = bbsNMS(:,11)';
    detScore{imgIdx} = bbsNMS(:,end)';

    for bbsIdx = 1:nDet
      if bbsNMS(bbsIdx,9) > 0.5 && ~detected
        tp{imgIdx}(bbsIdx) = 1;
        detected = true;
        if bbsNMS(bbsIdx,10) == 1
          atp{imgIdx}(bbsIdx) = 1;
        else
          afp{imgIdx}(bbsIdx) = 1;
        end
      else
        fp{imgIdx}(bbsIdx) = 1;
        afp{imgIdx}(bbsIdx) = 1;
      end
    end
    
    if visualize
      padding = 100;
      paddedIm = pad_image(im2double(im), padding, 1);
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
        resizeRendering = imresize(detectors{bbsNMS(bbsIdx, 11)}.rendering, [bnd(4) - bnd(2) + 1, bnd(3) - bnd(1) + 1]);
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
        titler = {['score ' num2str( bbsNMS(bbsIdx,12))], ...
          [' overlap ' num2str( bbsNMS(bbsIdx,9))],...
          [' azimuth D/GT ' num2str( detectors{bbsNMS(bbsIdx,11)}.az) ' ' num2str(azGT)],...
          [' azimuth ' num2str(bbsNMS(bbsIdx,10))]};

        plot_bbox(clip_bnd,cell2mat(titler),[1 1 1]);
      end
      drawnow;
      disp('Press any button to continue');
      waitforbuttonpress;
    end
      
    fprintf('%d/%d time : %0.4f\n', imgIdx, N_IMAGE, toc(imgTic));
end

detScore = cell2mat(detScore);
fp = cell2mat(fp);
tp = cell2mat(tp);
% atp = cell2mat(atp);
% afp = cell2mat(afp);
detectorId = cell2mat(detectorId);

[sc, si] =sort(detScore,'descend');
fpSort = cumsum(fp(si));
tpSort = cumsum(tp(si));

% atpSort = cumsum(atp(si));
% afpSort = cumsum(afp(si));

detectorIdSort = detectorId(si);

recall = tpSort/npos;
precision = tpSort./(fpSort + tpSort);

% arecall = atpSort/npos;
% aprecision = atpSort./(afpSort + atpSort);

ap = VOCap(recall', precision');
% aa = VOCap(arecall', aprecision');
fprintf('AP = %.4f\n', ap);
plot(recall, precision, 'r', 'LineWidth',3);
% hold on;
% plot(arecall, aprecision, 'g', 'LineWidth',3);
xlabel('Recall');
% ylabel('Precision/Accuracy');
% tit = sprintf('Average Precision = %.1f / Average Accuracy = %1.1f', 100*ap,100*aa);

tit = sprintf('Average Precision = %.1f', 100*ap);
title(tit);
axis([0 1 0 1]);
set(gcf,'color','w');
saveas(gcf,sprintf('Result/VOC_NCells_%d_lambda_%0.4f_azs_%d_els_%d_fovs_%d_honda_accord.png',...
  n_cell_limit,lambda,numel(azs),numel(els),numel(fovs)));