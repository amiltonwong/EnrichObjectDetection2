function [sprint_template, structure_data] = dwot_save_load_delegate(fid, save_format, image_name, structure_data) 
% delegate file for saving and loading detection results. See dwot_save_detection and dwot_load_detection for more detail
%   the save data structure must use thefollowing fields
%   prediction_scores, prediction_boxes, prediction_template_indexes, prediction_viewpoints, proposal_scores, proposal_boxes)

% If data comes, save it to file. Otherwise, load the data from the file
if nargin > 2
    [sprint_template] = get_sprint_template(save_format);
    [data_matrix] = get_structure_data(save_format, structure_data);
    for save_idx = 1:size(data_matrix,1)
        fprintf(fid, sprint_template, image_name, data_matrix(save_idx, :));
    end
else
    sprint_template = get_sprint_template(save_format);
    cell_data = textscan(fid, sprint_template);
    structure_data = convert_cell_data_to_structure_data(save_format, cell_data);
end



function [matrix_data] = get_structure_data(save_format, structure_data)

switch save_format
    case 'd'  
        % Detection only, 
        % filename, score, x1 y1 x2 y2
        matrix_data = [structure_data.prediction_scores, structure_data.prediction_boxes];
        assert(size(matrix_data,2) == 5);
    case 'dt'
        % Detection and template index
        % filename, score, x1 y1 x2 y2 ,template index
        matrix_data = [structure_data.prediction_scores, structure_data.prediction_boxes, structure_data.prediction_template_indexes];
        assert(size(matrix_data,2) == 6);
    case 'dv'
        % Detection and viewpoint
        % filename, score, x1 y1 x2 y2, viewpoint 
        matrix_data = [structure_data.prediction_scores, structure_data.prediction_boxes, structure_data.prediction_viewpoints];
        assert(size(matrix_data,2) == 6);
    case 'dtp'
        % Detection, template index, score and boundingbox from proposal detection
        % filename, score, x1 y1 x2 y2 ,template index, proposal score, proposal_x1, proposal_y1, proposal x2, proposal y2
        matrix_data = [structure_data.prediction_scores, structure_data.prediction_boxes, structure_data.prediction_template_indexes,...
            structure_data.proposal_scores, structure_data.proposal_boxes];
        assert(size(matrix_data,2) == 11);
    case 'dvp'
        % Detection, viewpoint, score and bounding box from proposal detection
        % filename, score, x1 y1 x2 y2 ,template index, proposal score, proposal_x1, proposal_y1, proposal x2, proposal y2
        % sprint_template = '%s %f %f %f %f %f %f %f %f %f %f %f';
        matrix_data = [structure_data.prediction_scores, structure_data.prediction_boxes, structure_data.prediction_viewpoints,...
                structure_data.proposal_scores, structure_data.proposal_boxes];
        assert(size(matrix_data,2) == 11);
    otherwise
        error('Undefied mode');
end 



function structure_data = convert_cell_data_to_structure_data(save_format, cell_data)
% scores, prediction_boxes, template_indexes, viewpoints, proposal_scores, proposal_boxes)

structure_data = [];
structure_data.file_names = cell_data{1};
structure_data.prediction_scores = cell_data{2};
structure_data.prediction_boxes = cell2mat(cell_data(3:6));
switch save_format
    case 'd'  
        % Detection only, 
        % filename, score, x1 y1 x2 y2
        % sprint_template = '%s %f %f %f %f %f';
    case 'dt'
        % Detection and template index
        % filename, score, x1 y1 x2 y2 ,template index
        % sprint_template = '%s %f %f %f %f %f %d';
        structure_data.prediction_template_indexes = cell_data{7};
    case 'dv'
        % Detection and viewpoint
        % filename, score, x1 y1 x2 y2, viewpoint 
        % sprint_template = '%s %f %f %f %f %f %f';
        structure_data.prediction_viewpoints = cell_data{7};
    case 'dtp'
        % Detection, template index, score and boundingbox from proposal detection
        % filename, score, x1 y1 x2 y2 ,template index, proposal score, proposal_x1, proposal_y1, proposal x2, proposal y2
        % sprint_template = '%s %f %f %f %f %f %d %f %f %f %f %f';
        structure_data.prediction_template_indexes = cell_data{7};
        structure_data.proposal_scores = cell_data{8};
        structure_data.proposal_boxes = cell2mat(cell_data(9:12));
    case 'dvp'
        % Detection, viewpoint, score and bounding box from proposal detection
        % filename, score, x1 y1 x2 y2 ,template index, proposal score, proposal_x1, proposal_y1, proposal x2, proposal y2
        % sprint_template = '%s %f %f %f %f %f %f %f %f %f %f %f';
        structure_data.prediction_viewpoints = cell_data{7};
        structure_data.proposal_scores = cell_data{8};
        structure_data.proposal_boxes = cell2mat(cell_data(9:12));
    otherwise
        error('Undefied mode');
end 



function [sprint_template] = get_sprint_template(save_format)

switch save_format
    case 'd'  
        % Detection only, 
        % filename, score, x1 y1 x2 y2
        sprint_template = '%s %f %f %f %f %f\n';
    case 'dt'
        % Detection and template index
        % filename, score, x1 y1 x2 y2 ,template index
        sprint_template = '%s %f %f %f %f %f %d\n';
    case 'dv'
        % Detection and viewpoint
        % filename, score, x1 y1 x2 y2, viewpoint 
        sprint_template = '%s %f %f %f %f %f %f\n';
    case 'dtp'
        % Detection, template index, score and boundingbox from proposal detection
        % filename, score, x1 y1 x2 y2 ,template index, proposal score, proposal_x1, proposal_y1, proposal x2, proposal y2
        sprint_template = '%s %f %f %f %f %f %d %f %f %f %f %f\n';

    case 'dvp'
        % Detection, viewpoint, score and bounding box from proposal detection
        % filename, score, x1 y1 x2 y2 ,template index, proposal score, proposal_x1, proposal_y1, proposal x2, proposal y2
        sprint_template = '%s %f %f %f %f %f %f %f %f %f %f %f\n';
    otherwise
        error('Undefied mode');
end 



