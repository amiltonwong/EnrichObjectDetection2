addpath('HoG');
addpath('HoG/features');
addpath('Util');
addpath('DecorrelateFeature/');
addpath('../MatlabRenderer/');
addpath('../MatlabRenderer/bin');
addpath('../MatlabCUDAConv/');
addpath('3rdParty/SpacePlot');
addpath('3rdParty/MinMaxSelection');
% addpath('Diagnosis');
rng('default');
DATA_SET = 'PASCAL12';
dwot_set_datapath;

COMPUTING_MODE = 1;
CLASS = 'Car';
SUB_CLASS = [];     % Sub folders
LOWER_CASE_CLASS = lower(CLASS);
TEST_TYPE = 'val';
SAVE_PATH = fullfile('Result',[LOWER_CASE_CLASS '_' TEST_TYPE]);
if ~exist(SAVE_PATH,'dir'); mkdir(SAVE_PATH); end

DEVICE_ID = 0; % 0-base indexing

if COMPUTING_MODE > 0
  gdevice = gpuDevice(DEVICE_ID + 1); % Matlab use 1 base indexing
  reset(gdevice);
  cos(gpuArray(1));
end
daz = 45;
del = 20;
dfov = 10;
dyaw = 10;

azs = 0:15:345; % azs = [azs , azs - 10, azs + 10];
els = 0:10:20;
fovs = [25];
yaws = 0;
n_cell_limit = [250];
lambda = [0.15];
detection_threshold = 80;

visualize_detection = true;
visualize_detector = false;

sbin = 6;
n_level = 20;
n_max_proposals = 10;
n_max_tuning = 1;
% Load models
% models_path = {'Mesh/Bicycle/road_bike'};
% models_name = cellfun(@(x) strrep(x, '/', '_'), models_path, 'UniformOutput', false);
[ model_names, model_paths ] = dwot_get_cad_models('Mesh', CLASS, [], {'3ds','obj'});

% models_to_use = {'bmx_bike',...
%               'fixed_gear_road_bike',...
%               'glx_bike',...
%               'road_bike'};

% models_to_use = {'2012-VW-beetle-turbo',...
%               'Kia_Spectra5_2006',...
%               '2008-Jeep-Cherokee',...
%               'Ford Ranger Updated',...
%               'BMW_X1_2013',...
%               'Honda_Accord_Coupe_2009',...
%               'Porsche_911',...
%               '2009 Toyota Cargo'};
% 
% use_idx = ismember(model_names,models_to_use);
% 
% model_names = model_names(use_idx);
% model_paths = model_paths(use_idx);

% skip_criteria = {'empty', 'truncated','difficult'};
skip_criteria = {'none'};
skip_name = cellfun(@(x) x(1), skip_criteria);

%%%%%%%%%%%%%%% Set Parameters %%%%%%%%%%%%
dwot_get_default_params;

param.template_initialization_mode = 0; 
param.nms_threshold = 0.4;
param.model_paths = model_paths;

param.b_calibrate = 0;      % apply callibration if > 0
param.n_calibration_images = 100; 
param.calibration_mode = 'gaussian';

param.detection_threshold = 30;
param.image_scale_factor = 2; % scale image accordingly and detect on the scaled image

% Tuning mode == 'none', no tuning
%             == 'mcm', MCMC
%             == 'not supported yet', Breadth first search
%             == 'bfgs', Quasi-Newton method (BFGS)
param.proposal_tuning_mode = 'none';

% Detection mode == 'dwot' ours
%                == 'cnn'
%                == 'dpm'
param.detection_mode = 'cnn';

% image_region_extraction.padding_ratio = 0.2;
param.PASCAL3D_ANNOTATION_PATH = PASCAL3D_ANNOTATION_PATH;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param.color_range = [-inf 20:5:100 inf];
param.cnn_color_range = [ -inf -4:0.1:3 inf];

% detector name
[ detector_model_name ] = dwot_get_detector_name(CLASS, SUB_CLASS, model_names, param);
detector_name = sprintf('%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d',...
            LOWER_CASE_CLASS,  detector_model_name, n_cell_limit, lambda,...
            numel(azs), numel(els), numel(yaws), numel(fovs));

detector_file_name = sprintf('%s.mat', detector_name);

%% Make empty detection save file
% if ~strcmp(param.detection_mode,'dwot')
detection_result_file = sprintf(['%s_%s_%s_%s_lim_%d_lam_%0.3f_a_%d_e_%d_y_%d_f_%d_scale_',...
            '%0.2f_sbin_%d_level_%d_server_%s.txt'],...
            DATA_SET, LOWER_CASE_CLASS, TEST_TYPE, detector_model_name, n_cell_limit, lambda,...
            numel(azs), numel(els), numel(yaws), numel(fovs), param.image_scale_factor, sbin,...
            n_level, server_id.num);

% Check duplicate file name and return different name
detection_result_file = dwot_save_detection([], SAVE_PATH, detection_result_file, [], true);
detection_result_common_name = regexp(detection_result_file, '\/?(?<name>.+)\.txt','names');
detection_result_common_name = detection_result_common_name.name;
fprintf('\nThe result will be saved on %s\n',detection_result_file);

% For detection right after the 
detection_dwot_proposal_result_file = sprintf(['%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_scale_',...
            '%0.2f_sbin_%d_level_%d_nms_%0.2f_server_%s_cnn_proposal_dwot.txt'],...
            DATA_SET, LOWER_CASE_CLASS, TEST_TYPE, detector_model_name, n_cell_limit, lambda,...
            numel(azs), numel(els), numel(yaws), numel(fovs), param.image_scale_factor, sbin,...
            n_level, param.nms_threshold, server_id.num);

% Check duplicate file name and return different name
detection_dwot_proposal_result_file = dwot_save_detection([], SAVE_PATH, detection_dwot_proposal_result_file, [], true);
fprintf('\nThe result will be saved on %s\n',detection_dwot_proposal_result_file);


% If tuning mode is defined and not none.
if isfield(param,'proposal_tuning_mode') && ~strcmp(param.proposal_tuning_mode,'none')
    detection_tuning_result_file = sprintf(['%s_%s_%s_%s_lim_%d_lam_%0.4f_a_%d_e_%d_y_%d_f_%d_scale_',...
            '%0.2f_sbin_%d_level_%d_nms_%0.2f_server_%s_tuning.txt'],...
            DATA_SET, LOWER_CASE_CLASS, TEST_TYPE, detector_model_name, n_cell_limit, lambda,...
            numel(azs), numel(els), numel(yaws), numel(fovs), param.image_scale_factor, sbin,...
            n_level, param.nms_threshold, server_id.num);
        
    % Check duplicate file name and return different name
    detection_tuning_result_file = dwot_save_detection([], SAVE_PATH, detection_tuning_result_file, [], true);
    detection_tuning_result_common_name = regexp(detection_tuning_result_file, '\/?(?<name>.+)\.txt','names');
    detection_tuning_result_common_name = detection_tuning_result_common_name.name;
    fprintf('\nThe tuning result will be saved on %s\n', detection_result_file);
end



%% Make Renderer
if ~strcmp(param.proposal_tuning_mode, 'none') || ~exist('renderer','var') || ~exist(detector_file_name,'file')
    % Initialize renderer
    renderer = Renderer();
    if ~renderer.initialize(model_paths, 700, 700, 0, 0, 0, 0, 25)
        error('fail to load model');
    end
end

%% Make Detectors

if exist(detector_file_name,'file')
    load(detector_file_name);
else
    [detectors] = dwot_make_detectors_grid(renderer, azs, els, yaws, fovs, 1:length(model_names),...
        LOWER_CASE_CLASS, param, visualize_detector);
    [detectors, detector_table]= dwot_make_table_from_detectors(detectors);
    if sum(cellfun(@(x) isempty(x), detectors))
      error('Detector Not Completed');
    end
    eval(sprintf(['save -v7.3 ' detector_file_name ' detectors detector_table']));
    % detectors = dwot_make_detectors(renderer, azs, els, yaws, fovs, param, visualize_detector);
    % eval(sprintf(['save ' detector_name ' detectors']));
end

%%%%% For Debugging
param.detectors              = detectors;
param.detect_pyramid_padding = 10;
%%%%%%%%%%%%

%% Make templates, these are just pointers to the templates in the detectors,
% The following code copies variables to GPU or make pointers to memory
% according to the computing mode.
%
% The GPU templates accelerates the computation time since it is already loaded
% on GPU.
if COMPUTING_MODE == 0
    % for CPU convolution, use fconvblas which handles template inversion
    templates_cpu = cellfun(@(x) single(x.whow), detectors,'UniformOutput',false);
elseif COMPUTING_MODE == 1
    % for GPU convolution, we use FFT based convolution. invert template
    templates_gpu = cellfun(@(x) gpuArray(single(x.whow(end:-1:1,end:-1:1,:))), detectors,'UniformOutput',false);
elseif COMPUTING_MODE == 2
    templates_gpu = cellfun(@(x) gpuArray(single(x.whow(end:-1:1,end:-1:1,:))), detectors,'UniformOutput',false);
    templates_cpu = cellfun(@(x) single(x.whow), detectors,'UniformOutput',false);
else
    error('Computing mode undefined');
end

template_size = cell2mat(cellfun(@(x) (x.sz)', detectors,'uniformoutput',false));
max_template_size = max(template_size, [],2);
min_template_size = min(template_size, [],2);


%% Load CNN Proposals
if ~exist('cnn_detection','var')
    cnn_detection = load('3dpascal_pascal12val_rcnn_detections');
    cnn_class_idx = find(strcmp(LOWER_CASE_CLASS, cnn_detection.classes));
end

[gtids,t] = textread(sprintf(VOCopts.imgsetpath,[LOWER_CASE_CLASS '_' TEST_TYPE]),'%s %d');

N_IMAGE = length(gtids);
% N_IMAGE = 1000;

clear gt;
gt = struct('BB',[],'diff',[],'det',[]);

for img_idx=1:N_IMAGE
    fprintf('%d/%d ',img_idx,N_IMAGE);
    % Find file name
    img_file_name = regexp(cnn_detection.imgFilePaths{img_idx}, '\/(?<img>\w+)\.png','names');
    img_file_name = img_file_name.img;

    imgTic = tic;
    % read annotation
    recs = PASreadrecord(sprintf(VOCopts.annopath, img_file_name));
    
    % Read 3D viewpoint annotation
    
    clsinds = find(strcmp(LOWER_CASE_CLASS, {recs.objects(:).class}));

    [skip_img, object_idx] = dwot_skip_criteria(recs.objects(clsinds), skip_criteria);
    
    if skip_img; continue; end;
    
    clsinds = clsinds(object_idx);
  
    switch param.detection_mode
      case 'dwot'
        if COMPUTING_MODE == 0
          [bbsAllLevel, hog, scales] = dwot_detect( im, templates_cpu, param);
    %       [hog_region_pyramid, im_region] = dwot_extract_region_conv(im, hog, scales, bbsNMS, param);
    %       [bbsNMS_MCMC] = dwot_mcmc_proposal_region(im, hog, scale, hog_region_pyramid, param);
        elseif COMPUTING_MODE == 1
          % [bbsNMS ] = dwot_detect_gpu_and_cpu( im, templates, templates_cpu, param);
          [bbsAllLevel, hog, scales] = dwot_detect_gpu( im, templates_gpu, param);
    %       [hog_region_pyramid, im_region] = dwot_extract_region_fft(im, hog, scales, bbsNMS, param);
        elseif COMPUTING_MODE == 2
          [bbsAllLevel, hog, scales] = dwot_detect_combined( im, templates_gpu, templates_cpu, param);
        else
          error('Computing Mode Undefined');
        end
        fprintf('convolution time: %0.4f\n', toc(imgTic));
      case 'cnn'
        bounding_box_proposals = cnn_detection.detBoxes{cnn_class_idx}{img_idx};
        bounding_box_proposals = bounding_box_proposals((bounding_box_proposals(:,end) > -inf ),:);
        bbsAllLevel = double(bounding_box_proposals);
        bbsAllLevel(:,1:4) = param.image_scale_factor * bbsAllLevel(:,1:4);
      case 'dpm'
        error('NOT SUPPORTED');
    end 
    
    
    gt.BB = param.image_scale_factor * cat(1, recs.objects(clsinds).bbox)';
    gt.diff = [recs.objects(clsinds).difficult];
    gt.det = zeros(length(clsinds),1);
    
    im = imread([VOCopts.datadir, recs.imgname]);
    im = imresize(im, param.image_scale_factor);
    im_size = size(im);
    
    bbsNMS = esvm_nms(bbsAllLevel, param.nms_threshold);
    % Automatically sort them according to the score and apply NMS
%     bbsNMS_clip = clip_to_image(bbsNMS, [1 1 im_size(2) im_size(1)]);
%     [ bbsNMS_clip, tp ] = dwot_compute_positives(bbsNMS_clip, gt, param);
    
    if numel(bbsNMS) == 0 
        temp = zeros(1,5);
        temp( end ) = -inf;
    else
        bbsNMS = bbsNMS((bbsNMS(:,end) > -1),:);
        if numel(bbsNMS) == 0 
            temp = zeros(1,5);
            temp( end ) = -inf;
        else
            temp = bbsNMS;
        end
    end
    
    dwot_save_detection(temp, SAVE_PATH, detection_result_file, ...
                                 img_file_name, false, 0); 
    
    [~, img_file_name] = fileparts(recs.imgname);
    % save mode != 0 to save template index
    
%     if visualize_detection && ~isempty(clsinds)
%         tpIdx = logical(tp); % Index of bounding boxes that will be printed as ground truth
%         try
%             clf;
%         end
%         dwot_visualize_result;
%         save_name = sprintf(['%s_img_%d.jpg'],...
%                            detection_result_common_name, img_idx);
%         print('-djpeg','-r150',fullfile(SAVE_PATH, save_name));
%     end
    
    %% Proposal Detection
    n_proposals = min([n_max_proposals, size(bbsNMS,1) ]);
    if  n_proposals > 0
        tuningTic = tic;

        bbs_tuning = cell(n_proposals , 1);
        
        % For each proposals draw original, before and after
        for proposal_idx = 1:n_proposals
            bbox = bbsNMS(proposal_idx, :);
            
            extraction_padding_ratio = 0.3;
            
            [proposal_im, clip_padded_bbox, clip_padded_bbox_offset, width, height] =...
                    dwot_extract_image_with_padding(im, bbox, extraction_padding_ratio, im_size);

            % Find right scale to search. Resize image if it's too small
            search_scale = max_template_size(1:2)' * sbin ./ [height width];
            proposal_resize_scale = 1;
            if max(search_scale) >= 0.7
                proposal_resize_scale = max(search_scale) * 1.6;
                proposal_im = imresize(proposal_im, proposal_resize_scale);
                search_scale = search_scale ./ proposal_resize_scale;
            end
            param.detect_min_scale = min(1, min(search_scale) / 1.6);
            param.detect_max_scale = min(1, min(search_scale) * 1.6);

            
            % Detect object and get initial proposal regions
            [bbsAllLevel, hog, scales] = dwot_detect_gpu( proposal_im, templates_gpu, param);
            
            if numel(bbsAllLevel) == 0; continue; end;
            
            bbsNMS_dwot_proposal = esvm_nms(bbsAllLevel,param.nms_threshold);
            
            % Save detection result
            save_bbs_nms = esvm_nms(bbsAllLevel,0.7);
            save_bbs_nms(:,1:4) = bsxfun(@plus, save_bbs_nms(:,1:4)/proposal_resize_scale, clip_padded_bbox_offset);
            dwot_save_detection(save_bbs_nms, SAVE_PATH, detection_dwot_proposal_result_file, ...
                                  img_file_name, false, 1); % save mode != 0 to save template index
                              
            n_detection_per_proposal = size( bbsNMS_dwot_proposal, 1);
            n_tuning_per_proposal = min(n_max_tuning, n_detection_per_proposal);

            % Fine tuning
            if strcmp(param.proposal_tuning_mode,'mcmc') 
                if n_tuning_per_proposal == 0; fprintf('no detection'); continue; end;

                [hog_region_pyramids, im_regions] = dwot_extract_hog(hog, scales, detectors, ...
                                                     bbsNMS_dwot_proposal(1:n_tuning_per_proposal ,:), param, proposal_im);
                [best_proposal_tunings] = dwot_mcmc_proposal_region(renderer, hog_region_pyramids, im_regions,...
                                                 detectors, param, im, false);

                fprintf(' tuning time : %0.4f\n', toc(tuningTic));

                % Sort the scores 
                [~, tuning_sorting_idx] = sort(cellfun(@(x) x.score, best_proposal_tunings),'descend');         
                best_proposal_tunings = best_proposal_tunings(tuning_sorting_idx);
                bbsNMS_dwot_proposal = bbsNMS_dwot_proposal(tuning_sorting_idx,:);
                bbs_tuning_per_proposal = zeros(n_tuning_per_proposal,12);

                % For each of the proposal 
                for detection_per_proposal_idx = n_tuning_per_proposal:-1:1

                    % fill out the box proposal infomation
                    bbs_tuning_per_proposal(detection_per_proposal_idx, 1:4) = best_proposal_tunings{detection_per_proposal_idx}.image_bbox;
                    bb_clip = clip_to_image(bbs_tuning_per_proposal(detection_per_proposal_idx, :), [1 1 im_size(2) im_size(1)]);
                    bb_clip = dwot_compute_positives(bb_clip, gt, param);
                    bbs_tuning_per_proposal(detection_per_proposal_idx, 9) = bb_clip(9);
                    bbs_tuning_per_proposal(detection_per_proposal_idx, 12) = best_proposal_tunings{detection_per_proposal_idx}.score;
                    bbs_tuning_per_proposal(detection_per_proposal_idx, 11) = 1;

                                      
                    bbs_temp = bbsNMS_dwot_proposal(detection_per_proposal_idx,:);
                    bbs_temp(1:4) = bbs_temp(1:4)/proposal_resize_scale + clip_padded_bbox_offset;
                    bbs_tuning_temp = bbs_tuning_per_proposal(detection_per_proposal_idx,:);
                    bbs_tuning_temp(1:4) = bbs_tuning_temp(1:4)/proposal_resize_scale + clip_padded_bbox_offset;

                    % Plot original image with GT bounding box
                    subplot(221);
                    dwot_visualize_result;
                    %     rectangle('position',dwot_bbox_xy_to_wh(GT_bbox),'edgecolor',[0.7 0.7 0.7],'LineWidth',3);
                    %     rectangle('position',dwot_bbox_xy_to_wh(GT_bbox),'edgecolor',[0   0   0.6],'LineWidth',2);

                    % Plot proposal bbox 
                    subplot(222);
                    dwot_draw_overlap_rendering(proposal_im, bbsNMS_dwot_proposal(detection_per_proposal_idx,:), detectors, n_tuning_per_proposal, 10,...
                                        visualize_detection, [0.3, 0.7, 0], param.color_range , 1 );

                    % dwot_draw_overlap_rendering(im, bbs_temp, detectors, 1, 50, true, [0.5, 0.5, 0] , param.color_range );
                    % axis equal; axis tight;

                    % Plot tuned bbox
                    subplot(223);
                    dwot_draw_overlap_rendering(proposal_im, bbs_tuning_per_proposal(detection_per_proposal_idx,:), {best_proposal_tunings{detection_per_proposal_idx}}, n_tuning_per_proposal, 10,...
                                        visualize_detection, [0.3, 0.7, 0], param.color_range , 1 );
                    % dwot_draw_overlap_rendering(im, bbsProposal, {best_proposal}, 1, 50, true, [0.1, 0.9, 0] , param.color_range );
                    % axis equal; axis tight;


                    subplot(224);
                    dwot_draw_overlap_rendering(im, bbs_tuning_temp, {best_proposal_tunings{detection_per_proposal_idx}}, n_tuning_per_proposal, 10,...
                                        visualize_detection, [0.3, 0.7, 0], param.color_range , 1 );

%                 dwot_draw_overlap_rendering(proposal_im, bbs_tuning_per_proposal(detection_per_proposal_idx,:),...
%                         {best_proposal_tunings{detection_per_proposal_idx}}, 1, 50, true, [0.3, 0.7, 0] , param.color_range );
%                 axis equal; axis tight;
%                 dwot_visualize_proposal_tuning(bbs_temp, bbs_tuning_temp, ...
%                                                 best_proposal_tunings{detection_per_proposal_idx}, im, detectors, param);
                    spaceplots();
                    drawnow;
                    save_name = sprintf(['%s_img_%d_prop_%d_%d.jpg'],...
                              detection_tuning_result_common_name, img_idx, proposal_idx, detection_per_proposal_idx);
                    print('-djpeg','-r150',fullfile(SAVE_PATH, save_name));
                end

                bbs_tuning{proposal_idx} = bbs_tuning_per_proposal;

                % Modify to save into valid image coord
                bbs_tuning{proposal_idx}(:,1:4) = bbs_tuning_per_proposal(:,1:4)/proposal_resize_scale + repmat(clip_padded_bbox_offset,n_tuning_per_proposal,1);
                bbs_tuning{proposal_idx}(:,12)  = bbs_tuning_per_proposal(:,end);
                             
            else
                % Draw original image and proposal region
                subplot(121);
                imagesc(im);
                dwot_draw_bounding_box(bbsNMS(proposal_idx,:), param);
                
                % Plot proposal bbox 
                subplot(122);
                dwot_draw_overlap_rendering(im, esvm_nms(save_bbs_nms, param.nms_threshold), detectors, 3, 0,...
                                    visualize_detection, [0.3, 0.7, 0], param.color_range , 1 );
                save_name = sprintf(['%s_img_%d_prop_%d.jpg'], detection_tuning_result_common_name,...
                                    img_idx, proposal_idx);
                print('-djpeg','-r150',fullfile(SAVE_PATH, save_name));
            end
        end

        % If tuning, save the final result after NMS
        if strcmp(param.proposal_tuning_mode,'mcmc') 
            bbs_tuning_mat_nms = esvm_nms(cell2mat(bbs_tuning), param.nms_threshold);
            if numel(bbs_tuning_mat_nms) == 0
                bbs_tuning_mat_nms = zeros(1,12);
                bbs_tuning_mat_nms(12) = -inf;
            end
            dwot_save_detection(bbs_tuning_mat_nms, SAVE_PATH, detection_tuning_result_file, ...
                                img_file_name, false, 1); % save mode != 0 to save template index
        end
    else
        % If there is no proposal detection make dummy detection and save
        bbs_tuning_mat_nms = zeros(1,12);
        bbs_tuning_mat_nms(12) = -inf;
        
        dwot_save_detection(bbs_tuning_mat_nms, SAVE_PATH, detection_dwot_proposal_result_file, ...
                                  img_file_name, false, 1);
                              
        if strcmp(param.proposal_tuning_mode,'mcmc') 
             dwot_save_detection(bbs_tuning_mat_nms, SAVE_PATH, detection_tuning_result_file, ...
                                         img_file_name, false, 1); % save mode != 0 to save template index
        end             
    end
end

close all;  % space plot casues problem when using different subplot grid

%% Vary NMS threshold
% for n_view = [4,8,16,24]
%     dwot_analyze_and_visualize_cnn_results( fullfile('Result/car_val', 'PASCAL12_car_val_init_0_Car_each_27_lim_250_lam_0.1500_a_24_e_3_y_1_f_1_scale_2.00_sbin_6_level_20_nms_0.30_server_104_cnn_proposal_dwot_tmp_2.txt') , detectors, '/home/chrischoy/DWOT_CNN5', VOCopts, param, skip_criteria, param.color_range, param.nms_threshold, false, n_view, -1, 0);
%     print('-dpng','-r200',fullfile(SAVE_PATH, ['AP_PASCAL12_car_val_init_0_Car_each_27_lim_250_lam_0.1500_a_24_e_3_y_1_f_1_scale_2.00_sbin_6_level_20_nms_0.30_server_104_cnn_proposal_dwot_tmp_2_nms_' sprintf('%0.2f',param.nms_threshold) '_nview_' num2str(n_view) '.png' ] ));
% end
nms_thresholds = 0.3:0.1:0.7;
ap = zeros(numel(nms_thresholds),1);
ap_dwot_cnn = zeros(numel(nms_thresholds),1);
ap_save_names = cell(numel(nms_thresholds),1);
ap_detection_save_names = cell(numel(nms_thresholds),1);
for i = 1:numel(nms_thresholds)
    nms_threshold = nms_thresholds(i);
    ap(i) = dwot_analyze_and_visualize_pascal_results(fullfile(SAVE_PATH, detection_result_file), ...
                        detectors, [], VOCopts, param, skip_criteria, param.color_range, ...
                        nms_threshold, false);
                        

    ap_save_names{i} = sprintf(['AP_%s_nms_%0.2f.png'],...
                        detection_result_common_name, nms_threshold);

     print('-dpng','-r150',fullfile(SAVE_PATH, ap_save_names{i}));
end


% Detection for regions before tuning
for i = 1:numel(nms_thresholds)
    nms_threshold = nms_thresholds(i);
    ap_dwot_cnn(i) = dwot_analyze_and_visualize_pascal_results(fullfile(SAVE_PATH,detection_dwot_proposal_result_file), ...
                        detectors, [], VOCopts, param, skip_criteria, param.color_range, ...
                        nms_threshold, false);
                        

    ap_detection_save_names{i} = sprintf(['AP_%s_cnn_proposal_dwot_nms_%0.2f.png'],...
                        detection_result_common_name, nms_threshold);

     print('-dpng','-r150',fullfile(SAVE_PATH, ap_detection_save_names{i}));
end


if ~strcmp(param.proposal_tuning_mode,'none')
    ap_tuning = dwot_analyze_and_visualize_pascal_results(fullfile(SAVE_PATH,...
                        detection_tuning_result_file), detectors, [], VOCopts, param,...
                        skip_criteria, param.color_range, param.nms_threshold, false);
                        
    ap_tuning_save_name = sprintf(['AP_%s_tuning_%s_nms_%.2f.png'],...
                        detection_result_common_name, param.proposal_tuning_mode,...
                        param.nms_threshold);

     print('-dpng','-r150',fullfile(SAVE_PATH, ap_tuning_save_name));
end

% If it runs on server copy to host
if ~isempty(server_id) && ~strcmp(server_id.num,'capri7')
    for i = 1:numel(nms_thresholds)
        system(['scp ', fullfile(SAVE_PATH, ap_save_names{i}),...
            ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/',...
            SAVE_PATH]);
        system(['scp ', fullfile(SAVE_PATH, ap_detection_save_names{i}),...
            ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/',...
            SAVE_PATH]);
    end
    system(['scp ' fullfile(SAVE_PATH, detection_result_file),...
        ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/' SAVE_PATH]);
    
    system(['scp ' fullfile(SAVE_PATH, detection_dwot_proposal_result_file),...
        ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/' SAVE_PATH]);
    
    if ~strcmp(param.proposal_tuning_mode,'none')
        system(['scp ', fullfile(SAVE_PATH, ap_tuning_save_name),...
            ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/',...
            SAVE_PATH]);
        system(['scp ' fullfile(SAVE_PATH, detection_tuning_result_file),...
            ' @capri7:/home/chrischoy/Dropbox/Research/DetectionWoTraining/',...
            SAVE_PATH]);
    end
end

