% compute recall and viewpoint accuracy
function [recall, precision, accuracy, ap, aa] = compute_recall_precision_accuracy_mod(cls, file_name, vnum, VOCopts, param, azimuth_interval)

if nargin < 6
    azimuth_interval = [0 (360/(vnum*2)):(360/vnum):360-(360/(vnum*2))];
end

% viewpoint annotation path
path_ann_view = '/home/chrischoy/Dataset/PASCAL3D+_release1.1/Annotations';

% read ids of validation images
ids = textread(sprintf(VOCopts.imgsetpath, 'val'), '%s');
M = numel(ids);

% open prediction file
object = load(file_name);
dets_all = object.res;
n_predictions = numel(object.detection);

energy = zeros(n_predictions,1);
tp = zeros(n_predictions,1);
fp = zeros(n_predictions,1);
tp_view = zeros(n_predictions, 1);
fp_view = zeros(n_predictions, 1);
% overlap = [];
count = zeros(M,1);
num = zeros(M,1);
num_pr = 0;
for i = 1:M
%     fprintf('%s view %d: %d/%d\n', cls, vnum, i, M);    
    % read ground truth bounding box
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    [~, img_name, ~] = fileparts(rec.filename);
    clsinds = strmatch(cls, {rec.objects(:).class}, 'exact');
    diff = [rec.objects(clsinds).difficult];
    % clsinds(diff == 1) = [];
    n = numel(clsinds);
    ground_truth_bboxes = zeros(n, 4);
    for j = 1:n
        ground_truth_bboxes(j,:) = rec.objects(clsinds(j)).bbox;
    end
    count(i) = size(ground_truth_bboxes, 1);
    det = zeros(count(i), 1);
    
    % read ground truth viewpoint
    if isempty(clsinds) == 0
        anno_name = fullfile(param.PASCAL3D_ANNOTATION_PATH,...
                sprintf('%s_pascal/%s.mat', cls, img_name));
        object = load(anno_name);
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
    dets = dets_all(i).detections;
    num(i) = size(dets, 2);
    % for each predicted bounding box
    for j = 1:num(i)
        num_pr = num_pr + 1;
        energy(num_pr) = dets(j).score;        
        bbox_pr = dets(j).bb;
        if vnum == 24
            view_pr = dets(j).vp;
        else
            view_pr = find_interval(dets(j).vp, azimuth_interval);
        end
        
        % compute box overlap
        if isempty(ground_truth_bboxes) == 0
            o = boxoverlap(ground_truth_bboxes, bbox_pr);
            [maxo, index] = max(o);
            if maxo >= 0.5 && det(index) == 0
                overlap{num_pr} = index;
                tp(num_pr) = 1;
                det(index) = 1;
                % check viewpoint
                if view_pr == view_gt(index)
                    correct_view(num_pr) = 1;
                else
                    correct_view(num_pr) = 0;
                end
            else
                overlap{num_pr} = [];
                tp(num_pr) = 0;
                correct_view(num_pr) = 0;
            end
        else
            overlap{num_pr} = [];
            tp(num_pr) = 0;
            correct_view(num_pr) = 0;
        end
    end
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
tp = tp(index);
correct_view = correct_view(index);
n = numel(threshold);
recall = zeros(n,1);
recall_view = zeros(n,1);
precision = zeros(n,1);
accuracy = zeros(n,1);
num_correct = 0;
num_correct_view = 0;
for i = 1:n
    % compute precision
    num_positive = i;
    num_correct = num_correct + tp(i);
    if num_positive ~= 0
        precision(i) = num_correct / num_positive;
    else
        precision(i) = 0;
    end
    
    % compute accuracy
    num_correct_view = num_correct_view + correct_view(i);
    if num_correct ~= 0
        accuracy(i) = num_correct_view / num_positive;
    else
        accuracy(i) = 0;
    end
    
    % compute recall
    recall(i) = num_correct / sum(count);
    recall_view(i) = num_correct_view / sum(count);
end


ap = VOCap(recall, precision);
fprintf('AP = %.4f\n', ap);

aa = VOCap(recall_view, accuracy);
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