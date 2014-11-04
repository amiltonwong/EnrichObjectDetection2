function detectors = dwot_calibrate_detectors(detectors, LOWER_CASE_CLASS, VOCopts, ...
    param, visualize)
% Calibrate the object detectors, from the negative images, we gather negative image
% patches and make false positive rate to be 0.01 percent

if nargin < 5
    visualize = false;
end

sbin = param.sbin;
n_detectors = numel(detectors);
detection_scores = cell(1, n_detectors);
templates = cellfun(@(x) single(x.whow), detectors, 'UniformOutput', false);
sz = cellfun(@(x) size(x), templates, 'UniformOutput',false);

% load dataset
[gtids,t] = textread(sprintf(VOCopts.imgsetpath,'train'),'%s %d');


%% Collect statistics
n_processed = 0;
for imgIdx = 1:numel(gtids)
    calTic = tic;
    recs(imgIdx)=PASreadrecord(sprintf(VOCopts.annopath,gtids{imgIdx}));
    clsinds = strmatch(LOWER_CASE_CLASS,{recs(imgIdx).objects(:).class},'exact');
    
    % Make sure that image of the same class does not go into the calibration
    if ~isempty(clsinds)
        continue;
    end
   
    im = imread([VOCopts.datadir, recs(imgIdx).imgname]);
    doubleIm = im2double(im);
    [hog, scales] = esvm_pyramid(doubleIm, param);
    
    maxTemplateHeight = max(cellfun(@(x) x(1), sz));
    maxTemplateWidth = max(cellfun(@(x) x(2), sz));
    
    minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), hog);
    hog = hog(minsizes >= param.min_hog_length);
    scales = scales(minsizes >= param.min_hog_length);
    bbsAll = cell(length(hog),1);
    
    for level = length(hog):-1:1
      HM = cudaConvolutionFFT(single(hog{level}), maxTemplateHeight, maxTemplateWidth, templates, param.cuda_conv_n_threads);
      hog_size = size(hog{level});
      for det_idx = 1:n_detectors
        hm = reshape(HM{det_idx}(4:hog_size(1)+floor(sz{det_idx}(1)/2), 4:hog_size(2)+floor(sz{det_idx}(2)/2)),1,[]);
        % detection_scores{det_idx}{numel(detection_scores{det_idx}) + 1} = randsample(hm, floor(numel(hm) * 0.1));  
        detection_scores{det_idx}{numel(detection_scores{det_idx}) + 1} = hm(:);  
      end
    end
    n_processed = n_processed + 1;
    
    if n_processed >= param.n_calibration_images
        break;
    end
    
    if mod(n_processed,10) == 0
        fprintf(' avg time: %0.3f\n', toc(calTic)/10);
        calTic = tic;
    else
        fprintf('.');
    end
end

if n_processed < param.n_calibration_images
    warning('Number of requested calibration images was larger than the dataset size');
end
  

% Compute the convolution score with mean HOG feature
for det_idx = 1:n_detectors
    
    switch param.calibration_mode
        case 'gaussian'
            % normalize score by normalizing the negative instances score distribution
            detection_scores{det_idx} = cell2mat(detection_scores{det_idx}');
            detectors{det_idx}.mean = mean(detection_scores{det_idx});
            detectors{det_idx}.var = var(detection_scores{det_idx});
            detectors{det_idx}.sigma = sqrt(detectors{det_idx}.var);
        case 'linear'
            % Seeing 3D Chair, CVPR 14 approach
            muSwapDim = permute(param.hog_mu, [2 3 1]);
            muProdTemplate = bsxfun(@times, templates{det_idx} , muSwapDim);
            muProdTemplate = sum(muProdTemplate(:));
            detection_scores{det_idx} = cell2mat(detection_scores{det_idx}');
            n_sample = numel(detection_scores{det_idx});

            if isfield(param,'percent_calibration_fp')
                percent_calibration_fp = param.percent_calibration_fp;
            else
                percent_calibration_fp = 0.01;
            end

            % use min/max algorithm to find top k scores without sorting
            top_scores = maxk(detection_scores{det_idx}, ceil(percent_calibration_fp / 100 * n_sample));
            t = [muProdTemplate, 1; top_scores(end), 1]\[-1; 0];
            detectors{det_idx}.a = t(1);
            detectors{det_idx}.b = t(2);
        otherwise
            error(['Calibration mode undefined : ' calibration_mode]);
    end
end

  
%%%%% Visualize %%%%
if visualize
    figure(1);
    color_map = jet(n_detectors);
    hist_range = -100:1:100;
    for det_idx = 1:n_detectors
        detection_score_temp = detection_scores{det_idx};
        count_per_bin = histc(detection_score_temp, hist_range);
        
        % Divide by area to normalize the disbribution
        plot(hist_range + 5, count_per_bin/trapz(hist_range, count_per_bin),'Color',color_map(det_idx,:));
        
        mean(detection_score_temp)
        hold on;
        drawnow;
        % waitforbuttonpress;
    end
    xlabel('Detection Score');
    ylabel('Probability');

    figure(2);
    hist_range = -3:0.05:1;
    for det_idx = 1:n_detectors
        detection_score_temp = ...
          dwot_calibrate_score(detection_scores{det_idx},...
              det_idx, detectors, param);
            
        count_per_bin = histc(detection_score_temp, hist_range);
        
        % Divide by area to normalize the disbribution
        plot(hist_range, count_per_bin/trapz(hist_range, count_per_bin),'Color',color_map(det_idx,:));
        
        hold on;
        drawnow;
        % waitforbuttonpress;
    end
    xlabel('Detection Score');
    ylabel('Probability');
    
    set(findall(gcf,'type','text'),'FontSize',15,'fontWeight','normal')
    % print('-dpng','-r100','before_4.png');
end

%     if COMPUTING_MODE == 0
%       [bbsAllLevel, hog, scales] = dwot_detect( im, templates_cpu, param);
%     elseif COMPUTING_MODE == 1
%       [bbsAllLevel, hog, scales] = dwot_detect_gpu( im, templates_gpu, param);
%     elseif COMPUTING_MODE == 2
%       [bbsAllLevel, hog, scales] = dwot_detect_combined( im, templates_gpu, templates_cpu, param);
%     else
%       error('Computing Mode Undefined');
%     end
%     fprintf(' time to convolution: %0.4f\n', toc(imgTic));
