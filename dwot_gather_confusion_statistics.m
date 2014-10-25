function [confusion_statistics] = dwot_gather_confusion_statistics(confusion_statistics, ground_truth_azimuth,...
            prediction_azimuth, gt_idx_of_prediction, n_azimuth_view, prediction_direction, prediction_offset)
        
if nargin < 6
    prediction_direction = 1;
    prediction_offset = 0;
end

% Assume that the bbsNMS has viewpoint overlap on the 9th column 
% and detector index on the 11th column and score on the 12th column
N_DETECTION = numel(prediction_azimuth);
viewpoint_azimuths = linspace(0, 360, n_azimuth_view+1);
viewpoint_azimuths = viewpoint_azimuths(1:end-1);
for det_idx = 1:N_DETECTION
  gt_id = gt_idx_of_prediction(det_idx);
  
  if gt_id > 0 % if gt_id is non zero
    prediction_viewpoint_index = viewpointIndex(prediction_azimuth(det_idx),...
                                viewpoint_azimuths, prediction_direction, prediction_offset);
    gt_viewpoint_index = viewpointIndex(ground_truth_azimuth(gt_id), viewpoint_azimuths, 1, 0);
    confusion_statistics(prediction_viewpoint_index, gt_viewpoint_index) = ...
                confusion_statistics(prediction_viewpoint_index, gt_viewpoint_index) + 1; 
  end
end

function viewpoint_index = viewpointIndex(query_azimuth, viewpoint_azimuths, detector_direction, gt_offset)
    % Helper function that determines the viewpoint index    
    % Compute viewpoint angle difference
    mod_az = mod( detector_direction * query_azimuth + gt_offset, 360 );
    [~, az_idx]  = min([abs(viewpoint_azimuths - mod_az),...
             abs(viewpoint_azimuths + 360 - mod_az),...
             abs(viewpoint_azimuths - 360 - mod_az)]);

    viewpoint_index = mod(az_idx-1,numel(viewpoint_azimuths)) + 1;
