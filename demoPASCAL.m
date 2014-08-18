VOC_PATH = '/home/chrischoy/Dataset/VOCdevkit/';
if ismac
  VOC_PATH = '~/dataset/VOCdevkit/';
end

addpath('HoG');
addpath('HoG/features');
addpath('Util');
addpath('../MatlabRenderer/');
addpath('../MatlabCUDAConv/');
addpath(VOC_PATH);
addpath([VOC_PATH, 'VOCcode']);

USE_GPU = true;
if USE_GPU
  gdevice = gpuDevice(1);
  reset(gdevice);
  cos(gpuArray(1));
end

CLASS = 'bicycle';
TYPE = 'val';
mkdir('Result',[CLASS '_' TYPE]);
% azs = 0:45:315; % azs = [azs , azs - 10, azs + 10];
% els = 0:30:30;
% fovs = [15, 45];
% yaws = [-30:30:30];
% n_cell_limit = [110];
% lambda = [0.015];

azs = 0:10:350
els = 0 : 10 : 40
fovs = [15, 45]
yaws = -40:10:40
n_cell_limit = 200
lambda = 0.015

visualize_detection = true;
visualize_detector = false;
% visualize = false;

sbin = 4;
nlevel = 10;
detection_threshold = 120;

model_file = 'Mesh/Bicycle/road_bike';
model_name = strrep(model_file, '/', '_');

detector_name = sprintf('%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d.mat',...
    model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs));

if exist(detector_name,'file')
  load(detector_name);
else
  load('Statistics/sumGamma_N1_40_N2_40_sbin_4_nLevel_10_nImg_1601_napoli1_gamma.mat');
  detectors = dwot_make_detectors_slow_gpu(mu, Gamma, [model_file '.3ds'], azs, els, yaws, fovs, n_cell_limit, lambda, visualize_detector);
  if sum(cellfun(@(x) isempty(x), detectors))
    error('Detector Not Completed');
  end
  eval(sprintf(['save ' detector_name ' detectors']));
end

renderings = cellfun(@(x) x.rendering, detectors, 'UniformOutput', false);

if USE_GPU
  templates = cellfun(@(x) gpuArray(single(x.whow(end:-1:1,end:-1:1,:))), detectors,'UniformOutput',false);
  % templates_cpu = cellfun(@(x) single(x.whow), detectors,'UniformOutput',false);
else
  templates = cellfun(@(x) single(x.whow), detectors,'UniformOutput',false);
end

param = dwot_get_default_params(sbin, nlevel, detection_threshold);
curDir = pwd;
eval(['cd ' VOC_PATH]);
VOCinit;
eval(['cd ' curDir]);


% load dataset
[gtids,t] = textread(sprintf(VOCopts.imgsetpath,[CLASS '_' TYPE]),'%s %d');

N_IMAGE = length(gtids);
N_IMAGE = 500;
% extract ground truth objects
npos = 0;
tp = cell(1,N_IMAGE);
fp = cell(1,N_IMAGE);
% atp = cell(1,N_IMAGE);
% afp = cell(1,N_IMAGE);
detScore = cell(1,N_IMAGE);
detectorId = cell(1,N_IMAGE);
detIdx = 0;


gt(length(gtids))=struct('BB',[],'diff',[],'det',[]);
for imgIdx=1:N_IMAGE
    fprintf('%d/%d ',imgIdx,N_IMAGE);
    imgTic = tic;
    % read annotation
    recs(imgIdx)=PASreadrecord(sprintf(VOCopts.annopath,gtids{imgIdx}));
    
    clsinds = strmatch(CLASS,{recs(imgIdx).objects(:).class},'exact');
    gt(imgIdx).BB=cat(1,recs(imgIdx).objects(clsinds).bbox)';
    gt(imgIdx).diff=[recs(imgIdx).objects(clsinds).difficult];
    gt(imgIdx).det=false(length(clsinds),1);
    
%     if isempty(clsinds)
%       continue;
%     end
    
    im = imread([VOCopts.datadir, recs(imgIdx).imgname]);
    imSz = size(im);
    if USE_GPU
      % [bbsNMS ] = dwot_detect_gpu_and_cpu( im, templates, templates_cpu, param);
      [bbsNMS ] = dwot_detect_gpu( im, templates, param);
    else
      [bbsNMS ] = dwot_detect( im, templates, param);
    end
    fprintf(' time to convolution: %0.4f', toc(imgTic));
        
    bbsNMS_clip = clip_to_image(bbsNMS, [1 1 imSz(2) imSz(1)]);

    nDet = size(bbsNMS_clip,1);
    tp{imgIdx} = zeros(1,nDet);
    fp{imgIdx} = zeros(1,nDet);
    
%     atp{imgIdx} = zeros(1,nDet);
%     afp{imgIdx} = zeros(1,nDet);

    if nDet > 0
      detectorId{imgIdx} = bbsNMS_clip(:,11)';
      detScore{imgIdx} = bbsNMS_clip(:,end)';
    else
      detectorId{imgIdx} = [];
      detScore{imgIdx} = [];
    end
    
    for bbsIdx = 1:nDet
      ovmax=-inf;
      
      % search over all objects in the image
      for j=1:size(gt(imgIdx).BB,2)
          bbgt=gt(imgIdx).BB(:,j);
          bi=[max(bbsNMS_clip(bbsIdx,1),bbgt(1)) ; max(bbsNMS_clip(bbsIdx,2),bbgt(2)) ; min(bbsNMS_clip(bbsIdx,3),bbgt(3)) ; min(bbsNMS_clip(bbsIdx,4),bbgt(4))];
          iw=bi(3)-bi(1)+1;
          ih=bi(4)-bi(2)+1;
          if iw>0 && ih>0                
              % compute overlap as area of intersection / area of union
              ua=(bbsNMS_clip(bbsIdx,3)-bbsNMS_clip(bbsIdx,1)+1)*(bbsNMS_clip(bbsIdx,4)-bbsNMS_clip(bbsIdx,2)+1)+...
                 (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
                 iw*ih;
              ov=iw*ih/ua;
              
              if ov > ovmax
                  ovmax = ov;
                  jmax = j;
              end
          end
      end
      
      % assign detection as true positive/don't care/false positive
      if ovmax >= VOCopts.minoverlap
          if ~gt(imgIdx).diff(jmax)
              if ~gt(imgIdx).det(jmax)
                  tp{imgIdx}(bbsIdx)=1;            % true positive
                  gt(imgIdx).det(jmax)=true;
              else
                  fp{imgIdx}(bbsIdx)=1;            % false positive (multiple detection)
              end
          end
      else
          fp{imgIdx}(bbsIdx)=1;                    % false positive
      end
      
      bbsNMS(bbsIdx, 9) = ovmax;
      bbsNMS_clip(bbsIdx, 9) = ovmax;
    end
    
    fprintf(' time : %0.4f\n', toc(imgTic));

    % if visualize
    if visualize_detection && ~isempty(clsinds)
      dwot_draw_overlap_detection(im, bbsNMS, renderings, 5, 50, visualize_detection);
      
      disp('Press any button to continue');
      
%       save_name = sprintf('%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_imgIdx_%d.jpg',...
%         CLASS,TYPE,model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs),imgIdx);
%       print('-djpeg','-r100',['Result/' CLASS '_' TYPE '/' save_name])
      
      % waitforbuttonpress;
    end
      
    npos=npos+sum(~gt(imgIdx).diff);
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

clf;
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
save_name = sprintf('AP_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d.jpg',...
        CLASS, TYPE, model_name, n_cell_limit, lambda, numel(azs), numel(els), numel(yaws), numel(fovs));

print('-dpng','-r150',['Result/' CLASS '_' TYPE '/' save_name])