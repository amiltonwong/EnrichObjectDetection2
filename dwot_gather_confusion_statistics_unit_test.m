function [success] = dwot_gather_confusion_statistics_unit_test

n_azimuth_views = 8;
confusion_statistics = zeros(n_azimuth_views, n_azimuth_views);

% ground truth
ground_truth_azimuth = [45, 10, 0];

% rotation direction, 1 if the prediction rotates the same clck wise or
% ctr-clck wise of the ground truth -1 otherwise
prediction_direction = 1;
prediction_offset = 90; % degree offset of the ground truth and the prediction viewpoint


%% First Round
% predicted azimuth
prediction_azimuth = [-45];

% index of ground truth that corresponds to the prediction
gt_idx_of_prediction = [ 1 ];

[confusion_statistics] = dwot_gather_confusion_statistics(confusion_statistics, ...
                    ground_truth_azimuth, prediction_azimuth, gt_idx_of_prediction, ...
                    n_azimuth_views, prediction_direction, prediction_offset );
                
assert(confusion_statistics(2,2) == 1);


%% Second Round
% predicted azimuth
prediction_azimuth = [-450];

% index of ground truth that corresponds to the prediction
gt_idx_of_prediction = [ 2 ];

[confusion_statistics] = dwot_gather_confusion_statistics(confusion_statistics, ...
                    ground_truth_azimuth, prediction_azimuth, gt_idx_of_prediction, ...
                    n_azimuth_views, prediction_direction, prediction_offset );

assert(confusion_statistics(1,1) == 1);


%% Second Round
% predicted azimuth
prediction_azimuth = [0];

% index of ground truth that corresponds to the prediction
gt_idx_of_prediction = [-1];

confusion_statistics_before = confusion_statistics;
[confusion_statistics_after] = dwot_gather_confusion_statistics(confusion_statistics, ...
                    ground_truth_azimuth, prediction_azimuth, gt_idx_of_prediction, ...
                    n_azimuth_views, prediction_direction, prediction_offset );
assert( prod(prod((confusion_statistics_after==confusion_statistics_before))) == 1 );
                
success = true;