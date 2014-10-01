function [confusion_statistics] = dwot_gather_confusion_statistics(confusion_statistics, detectors, GTs, bbsNMS, n_azimuth_view)
    % Assume that the bbsNMS has viewpoint overlap on the 9th column 
    % and detector index on the 11th column and score on the 12th column
    N_DETECTION = size(bbsNMS,1);
    viewpoint_azimuths = linspace(0, 360, n_azimuth_view+1);
    viewpoint_azimuths = viewpoint_azimuths(1:end-1);
    for det_idx = 1:N_DETECTION
      template_idx = bbsNMS(det_idx,11);
      gt_id = bbsNMS(det_idx,10);
      
      if gt_id % if gt_id is non zero
        prediction_viewpoint_index = viewpointIndex(detectors{template_idx}.az, viewpoint_azimuths);
        if iscell(GTs)
            gt_viewpoint_index = viewpointIndex(GTs{gt_id}.azimuth, viewpoint_azimuths);
        else
            gt_viewpoint_index = viewpointIndex(GTs(gt_id).azimuth, viewpoint_azimuths);
        end
        confusion_statistics(prediction_viewpoint_index, gt_viewpoint_index) = confusion_statistics(prediction_viewpoint_index, gt_viewpoint_index) + 1; 
      end
    end

function viewpoint_index = viewpointIndex(query_azimuth, viewpoint_azimuths)
% Helper function that determines the viewpoint index    
    [~, az_idx]  = min([abs(viewpoint_azimuths - query_azimuth),...
                   abs(viewpoint_azimuths + 360 - query_azimuth),...
                   abs(viewpoint_azimuths - 360 - query_azimuth)]);

    viewpoint_index = mod(az_idx-1,numel(viewpoint_azimuths)) + 1;
