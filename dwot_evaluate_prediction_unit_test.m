function success = dwot_evaluate_prediction_unit_test()

success = false;

ground_truth_bbox = [100 200 200 300;...
                     200 300 300 400];


%% First Round
prediction_bbox = [100 200 200 300;...
                   200 300 300 400];

[tp, fp, iou, corresponding_gt_idx] = dwot_evaluate_prediction(...
            prediction_bbox, ground_truth_bbox);

assert(logical(prod(tp == [true true])));
assert(logical(prod(fp == [false false])));
assert(logical(prod(corresponding_gt_idx == [1 2])));


%% Second Round
prediction_bbox = [50 200 200 300;...
                   300 300 400 400];

[tp, fp, iou, corresponding_gt_idx] = dwot_evaluate_prediction(...
            prediction_bbox, ground_truth_bbox);

assert(logical(prod(tp == [true false])));
assert(logical(prod(fp == [false true])));
assert(logical(prod(corresponding_gt_idx == [1 0])));


%% Third round
ground_truth_azimuth = [0; 240];

prediction_bbox = [50 200 200 300;... % overlaps exactly with first gt 
                   50 200 200 300;... % overlaps exactly with first gt 
                   200 300 300 400]; % overlaps exactly with second gt
prediction_azimuth = [45 0 240];  % viewpoint

exclude_ground_truth = [false true]; % second ground truth is not included
                                     % in the evaluation

[tp, fp, iou, corresponding_gt_idx] = dwot_evaluate_prediction(...
            prediction_bbox, ground_truth_bbox, 0.5,...
            exclude_ground_truth,...
            prediction_azimuth, ground_truth_azimuth);
        
assert(logical(prod(tp == [false true false])));
assert(logical(prod(fp == [true false false])));
assert(logical(prod(corresponding_gt_idx == [0 1 0])));


assert(prod(tp == [0 0]));
assert(prod(fp == [1 1]));
assert(prod(ov == [0 1]));
assert(prod(gt_idx_of_prediction == [-1 -1]));

success = true;
