function save_structure = dwot_formatted_bounding_boxes_to_save_structure(formatted_bounding_boxes, proposal_formatted_bounding_boxes)

% From dwot_formatted_bounding_boxes_to_predictions.m    
% 1:4 prediction_boxes
% 5   hog scale of the prediction, Not Used
% 6   hog pyramid level, Not Used
% 7   y_coord in hog pyramid, Not Used
% 8   x_coord in hog pyramid, Not Used
% 9   overlaps
% 10  viewpoint
% 11  detection template_indexes
% 12  score
save_structure = [];
if isempty(formatted_bounding_boxes)
    return;
end
save_structure.prediction_boxes            = formatted_bounding_boxes(:,1:4);
save_structure.prediction_viewpoints       = formatted_bounding_boxes(:, 10);
save_structure.prediction_template_indexes = formatted_bounding_boxes(:,11);
save_structure.prediction_scores           = formatted_bounding_boxes(:,end);
if nargin == 2
    save_structure.proposal_boxes          = proposal_formatted_bounding_boxes(:,1:4);
    save_structure.proposal_scores         = proposal_formatted_bounding_boxes(:,end);
end
