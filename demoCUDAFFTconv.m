% VOC_PATH = '/home/chrischoy/Dataset/VOCdevkit/';
clear;
addpath('HoG');
addpath('HoG/features');
addpath('Util');
addpath('../MatlabRenderer/');
addpath('../MatlabCUDAConv/');

IMAGE_PATH = '../PBR_MATLAB/OBJECT_3D/';
IMAGE_START_IDX = 121;
IMAGE_END_IDX = 240;
N_IMAGE = IMAGE_END_IDX - IMAGE_START_IDX + 1;

azs = 0:30:330; % azs = [azs , azs - 10, azs + 10];
els = [0:15:45];
fovs = [25];
yaws = ismac * 180 + (-45:15:45);
n_cell_limit = [150];
lambda = [0.02];
visualize = true;
% visualize = false;
sbin = 2;
nlevel = 10;
detection_threshold = 100;
HOGDim = 31;

model_file = 'Mesh/Bicycle/road_bike';
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

templates = cellfun(@(x) single(x.whow), detectors,'UniformOutput',false);
db_templates = cellfun(@(x) x.whow, detectors,'UniformOutput',false);

nTemplates =  numel(templates);
templateSz = cellfun(@(x) size(x), templates, 'UniformOutput',false);
templateGPU = cellfun(@(x) gpuArray(x(end:-1:1,end:-1:1,:)), templates, 'UniformOutput',false);

maxTemplateHeight = max(cellfun(@(x) x(1), templateSz));
maxTemplateWidth = max(cellfun(@(x) x(2), templateSz));
    
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

cos(gpuArray(1));

startTime = clock;
for imgIdx = 1:(IMAGE_END_IDX - IMAGE_START_IDX + 1)
    imgTic = tic;
    % read annotation
    annotation = load([IMAGE_PATH '/Annotation/' sprintf('%04d',imgIdx + IMAGE_START_IDX - 1) '.mat']);
    annotation = annotation.annotation;

    azGT = annotation.view(1);
    elGT = annotation.view(2);
    
    im = imread([IMAGE_PATH '/Images/' sprintf('%04d',imgIdx + IMAGE_START_IDX - 1) '.jpg']);
    
    
    doubleIm = im2double(im);
    [hog, scales] = esvm_pyramid(doubleIm, param);
    padder = param.detect_pyramid_padding;
    sbin = param.sbin;


    for level = 1:length(hog)
        hog{level} = padarray(hog{level}, [padder padder 0], 0); % Convolution, same size
    end

    minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), hog);
    hog = hog(minsizes >= padder*2);
    scales = scales(minsizes >= padder*2);
    bbsAll = cell(length(hog),1);

    % for level = length(hog):-1:1
    
    for level = 1:length(hog)
      singleHOG = single(hog{level});
      
      tic
      HM = fconvblasfloat(singleHOG, templates, 1, nTemplates);
      toc
      
      szHOG = size(hog{level});
      
      nlen = szHOG(1) + maxTemplateHeight;
      mlen = szHOG(2) + maxTemplateWidth;
      
      tic
      fhog = cudaFFTData(single(hog{level}), maxTemplateHeight, maxTemplateWidth);
%       [x, y] = ndgrid(1:nlen,1:mlen);
%       [xf, yf] = ndgrid(1:0.5:nlen,1:0.5:mlen);
%       vq = interpn(x,y,fftshift(real(fhog(:,:,1))),xf,yf,'cubic');
%       for templateIdx = 1:nTemplates
%         GPUTemplates{templateIdx} = gpuArray(single(templates{templateIdx}));
%       end
      
%       for templateIdx = 1:nTemplates
%         fthog{templateIdx} = zeros(nlen, mlen, HOGDim,'single','gpuArray');
%         for dIdx = 1:HOGDim
%           fthog{templateIdx}(:,:,dIdx) = conj(fftn(gpuArray(single(templates{templateIdx}(:,:,dIdx))), [nlen, mlen]));
%         end
%       end
      

      HMFFT = cell(1, nTemplates);
      for templateIdx = 1:nTemplates
        HMFFT{templateIdx} = cudaConvFFTData(fhog,templateGPU{templateIdx});
      end
      toc
%       db_cvmatlab = {};
%       for templateIdx = 1:nTemplates
%         szTemplate = size(db_templates{templateIdx});
%         e = zeros(szHOG(1) + szTemplate(1) - 1, szHOG(2) + szTemplate(2) - 1, HOGDim);
%         for i = 1:HOGDim
%           e(:,:,i) = conv2(hog{level}(:,:,i),db_templates{templateIdx}(end:-1:1,end:-1:1,i));
%         end
%         db_cvmatlab{templateIdx} = sum(e,3);
%       end
%       
%       
%       for templateIdx = 1:nTemplates
%         subplot(131); imagesc(HM{templateIdx}); caxis([-100 100]); colorbar; 
%         subplot(132); imagesc(HMFFT{templateIdx}); caxis([-100 100]); colorbar;
%         subplot(133); imagesc(db_cvmatlab{templateIdx}); caxis([-100 100]); colorbar;
%         waitforbuttonpress;
%       end

      scale = scales(level);
      
      rmsizes = cellfun(@(x) size(x), HM, 'UniformOutput',false);
      templateIdxes = find(cellfun(@(x) prod(x), rmsizes));
      bbsTemplate = cell(nTemplates,1);

      for templateIdx = templateIdxes
        [idx] = find(HM{templateIdx}(:) > param.detection_threshold);
        if isempty(idx)
          continue;
        end

        [uus,vvs] = ind2sub(rmsizes{templateIdx}(1:2), idx);

        o = [uus vvs] - padder;

        bbs = ([o(:,2) o(:,1) o(:,2)+templateSz{templateIdx}(2) ...
                   o(:,1)+templateSz{templateIdx}(1)] - 1) * ...
                 sbin/scale + 1 + repmat([0 0 -1 -1],...
                  length(uus),1);

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
      padding = 50;
      paddedIm = pad_image(im2double(im), padding, 1);
      resultIm = paddedIm;
      NDrawBox = min(nDet,4);
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
