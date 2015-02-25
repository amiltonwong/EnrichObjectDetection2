function [detector_indexes] = dwot_detector_kdtree_query(detectors_kdtree, az_range, el_range, yaw_range, fov_range, model_indexes, model_class, param)
% Given query ranges, divide the ranges so that it is circular.

az_valid_ranges = make_range_valid_degree(az_range);
el_valid_ranges = make_range_valid_degree(el_range);
% yaw_valid_ranges = make_range_valid_degree(az_range);
% fov_valid_ranges = make_range_valid_degree(az_range);

detector_indexes_for_range = cell(1, size(az_valid_ranges,1) * size(el_valid_ranges,1) );
range_index = 1;
for az_range_index = 1:size(az_valid_ranges,1)
    for el_range_index = 1:size(el_valid_ranges,1)
        viewpoint_range = [ az_valid_ranges(az_range_index, :);...
                            el_valid_ranges(el_range_index, :);...
                            yaw_range,...
                            fov_range];
        detector_indexes_for_range{range_index} = kdtree_range_query( detectors_kdtree, viewpoint_range );
        range_index = range_index + 1;
    end
end
detector_indexes = cell2mat(detector_indexes_for_range);

        

function valid_ranges = make_range_valid_degree(in_range)
% input : one contiguous range in degree
% output : set of contiguous ranges that represent input range 

% Assume that the upper limit is on the second column and the lower limit on the first column
% range_upper_lim = max(in_range);
% range_lower_lim = min(in_range);

valid_ranges = in_range;

break_upper_limit = false;

if range_upper_lim > 360
    valid_ranges(1,2) = 360;
    valid_ranges(2,1) =     in_range(1);
    valid_ranges(2,2) = mod(in_range(2), 360);

    break_upper_limit = true;
end

if range_lower_lim < 0
    valid_range(1,1) = 0;
    if break_upper_limit
        valid_range(2,1) = 0;
        valid_range(3,1) = 360 + in_range(1);
        valid_range(3,2) = 360;
    else
        valid_range(2,1) = 360 + in_range(1);
        valid_range(2,2) = 360;
    end
end
