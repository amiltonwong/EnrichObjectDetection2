% compute recall and viewpoint accuracy
function [recall, precision, accuracy, ap, aa] = compute_recall_precision_accuracy(detection_result_txt, vnum, VOCopts, param, azimuth_interval)

if nargin < 5
    azimuth_interval = [0 (360/(vnum*2)):(360/vnum):360-(360/(vnum*2))];
end

detection_params = dwot_detection_params_from_name(detection_result_txt);
image_scale_factor = detection_params.scale;
cls = detection_params.LOWER_CASE_CLASS;

% SNames = fieldnames(temp); 
% for loopIndex = 1:numel(SNames) 
%     stuff = S.(SNames{loopIndex})
% end

skip_criteria = dwot_recover_skip_criteria(detection_params.skip_name);

detection_param_name = regexp(detection_result_txt,'\/?([-\w\.]+)\.txt','tokens');
detection_name = detection_param_name{1}{1};

fileID = fopen(detection_result_txt,'r');
[~, detection_result] = dwot_save_load_delegate(fileID, detection_params.save_mode);
fclose(fileID);

[unique_files, ~, unique_file_idx] = unique(detection_result.file_names);
n_unique_files = numel(unique_files);

energy = [];
correct = [];
correct_view = [];
overlap = [];
count = zeros(n_unique_files,1);
num = zeros(n_unique_files,1);
num_pr = 0;

for prediction_idx = 1:n_unique_files
%     fprintf('%s view %d: %d/%d\n', cls, vnum, i, M);    
    % read ground truth bounding box
    rec = PASreadrecord(sprintf(VOCopts.annopath,unique_files{prediction_idx}));
    [~, file_name, ~] = fileparts(rec.filename);
    
    clsinds = strmatch(cls, {rec.objects(:).class}, 'exact');
    diff = [rec.objects(clsinds).difficult];
    clsinds(diff == 1) = [];
    n = numel(clsinds);
    bbox = zeros(n, 4);
    for j = 1:n
        bbox(j,:) = rec.objects(clsinds(j)).bbox;
    end
    count(prediction_idx) = size(bbox, 1);
    det = zeros(count(prediction_idx), 1);
    
    % read ground truth viewpoint
    if isempty(clsinds) == 0
        filename = fullfile(param.PASCAL3D_ANNOTATION_PATH,...
                sprintf('%s_pascal/%s.mat', detection_params.LOWER_CASE_CLASS, file_name));
        object = load(filename);
        record = object.record;
        view_gt = zeros(n, 1);
        for j = 1:n
            if record.objects(clsinds(j)).viewpoint.distance == 0
                azimuth = record.objects(clsinds(j)).viewpoint.azimuth_coarse;
            else
                azimuth = record.objects(clsinds(j)).viewpoint.azimuth;
            end
            view_gt(j) = find_interval(azimuth, azimuth_interval);
        end
    else
        view_gt = [];
    end
    
    % get predicted bounding box
    
    curr_file_idx = find(unique_file_idx == prediction_idx);
    
    prediction_bounding_boxes = detection_result.proposal_boxes(curr_file_idx,:)/image_scale_factor;
    prediction_scores = detection_result.proposal_scores(curr_file_idx,:);
    prediction_viewpoints =  detection_result.proposal_viewpoints(curr_file_idx,:);
    
    num(prediction_idx) = numel(prediction_scores);
    % for each predicted bounding box
    for j = 1:num(prediction_idx)
        num_pr = num_pr + 1;
        energy(num_pr) = prediction_scores(j);        
        bbox_pr = prediction_bounding_boxes(j,:);
        if vnum == 24
            view_pr = prediction_viewpoints(j);
        else
            view_pr = find_interval(prediction_viewpoints(j), azimuth_interval);
        end
        
        % compute box overlap
        if isempty(bbox) == 0
            o = boxoverlap(bbox, bbox_pr);
            [maxo, index] = max(o);
            if maxo >= 0.5 && det(index) == 0
                overlap{num_pr} = index;
                correct(num_pr) = 1;
                det(index) = 1;
                % check viewpoint
                if view_pr == view_gt(index)
                    correct_view(num_pr) = 1;
                else
                    correct_view(num_pr) = 0;
                end
            else
                overlap{num_pr} = [];
                correct(num_pr) = 0;
                correct_view(num_pr) = 0;
            end
        else
            overlap{num_pr} = [];
            correct(num_pr) = 0;
            correct_view(num_pr) = 0;
        end
    end
%     prediction_idx
%     correct_view(numel(correct_view)-num(prediction_idx)+1:end)
end
overlap = overlap';

% threshold = sort(energy);
% n = numel(threshold);
% recall = zeros(n,1);
% precision = zeros(n,1);
% accuracy = zeros(n,1);
% for i = 1:n
%     % compute precision
%     num_positive = numel(find(energy >= threshold(i)));
%     num_correct = sum(correct(energy >= threshold(i)));
%     if num_positive ~= 0
%         precision(i) = num_correct / num_positive;
%     else
%         precision(i) = 0;
%     end
%     
%     % compute accuracy
%     num_correct_view = sum(correct_view(energy >= threshold(i)));
%     if num_correct ~= 0
%         accuracy(i) = num_correct_view / num_positive;
%     else
%         accuracy(i) = 0;
%     end
%     
%     % compute recall
%     recall(i) = num_correct / sum(count);
% end
% 
% ap = VOCap(recall(end:-1:1), precision(end:-1:1));
% fprintf('AP = %.4f\n', ap);
% 
% aa = VOCap(recall(end:-1:1), accuracy(end:-1:1));
% fprintf('AA = %.4f\n', aa);

[threshold, index] = sort(energy, 'descend');
correct = correct(index);
correct_view = correct_view(index);
n = numel(threshold);
recall = zeros(n,1);
recall_view = zeros(n,1);
precision = zeros(n,1);
accuracy = zeros(n,1);
num_correct = 0;
num_correct_view = 0;
for prediction_idx = 1:n
    % compute precision
    num_positive = prediction_idx;
    num_correct = num_correct + correct(prediction_idx);
    if num_positive ~= 0
        precision(prediction_idx) = num_correct / num_positive;
    else
        precision(prediction_idx) = 0;
    end
    
    % compute accuracy
    num_correct_view = num_correct_view + correct_view(prediction_idx);
    if num_correct ~= 0
        accuracy(prediction_idx) = num_correct_view / num_positive;
    else
        accuracy(prediction_idx) = 0;
    end
    
    % compute recall
    recall(prediction_idx)          = num_correct / sum(count);
    recall_view(prediction_idx) = num_correct_view / sum(count);
end


ap = VOCap(recall, precision);
fprintf('AP = %.4f\n', ap);

aa = VOCap(recall, accuracy);
fprintf('AVP = %.4f\n', aa);

% draw recall-precision and accuracy curve
% figure;
% hold on;
% plot(recall, precision, 'r', 'LineWidth',3);
% plot(recall, accuracy, 'g', 'LineWidth',3);
% xlabel('Recall');
% ylabel('Precision/Accuracy');
% tit = sprintf('Average Precision = %.1f / Average Accuracy = %.1f', 100*ap, 100*aa);
% title(tit);
% hold off;


function ind = find_interval(azimuth, a)

for i = 1:numel(a)
    if azimuth < a(i)
        break;
    end
end
ind = i - 1;
if azimuth > a(end)
    ind = 1;
end