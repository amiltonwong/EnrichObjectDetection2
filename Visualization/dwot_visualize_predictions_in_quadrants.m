function dwot_visualize_predictions_in_quadrants(im, formatted_bounding_boxes, ground_truth_bounding_boxes, detectors, param)

if isfield(param,'text_mode')
    text_mode = param.text_mode;
else
    text_mode = 1;
end

if isfield(param,'color_range')
    color_range = param.color_range;
    if isfield(param,'color_map');
        color_map = param.color_map;
    else
        color_map = jet(numel(color_range));
    end  
else
    color_range = [-inf inf];
    color_map = cool(2);
end

if isfield(param,'n_max_fals_positive_visualization');
    n_max_fals_positive_visualization = param.n_max_fals_positive_visualization;
else
    n_max_fals_positive_visualization = 6;
end
n_prediction = size(formatted_bounding_boxes,1);
draw_padding = 0;

if n_prediction == 0
    warning('dwot_visualize_predictions_in_quadrants: no prediction to visualize');
    return;
end
[true_positive, false_positive, prediction_iou, ~] =...
    dwot_evaluate_prediction(...
            dwot_clip_bounding_box(formatted_bounding_boxes, size(im)),...
            ground_truth_bounding_boxes, param.min_overlap);

formatted_bounding_boxes(:,9) = prediction_iou;



% Sort the image according to it width
% the dwot_draw_overlap_rendering will draw from the last box so put the largest object
% first
formatted_bounding_boxes_tp = formatted_bounding_boxes(true_positive,:);
widths = formatted_bounding_boxes_tp (:,3) - formatted_bounding_boxes_tp (:,1);
[~, width_sort_idx ] = sort(widths, 'descend');
formatted_bounding_boxes_tp = formatted_bounding_boxes_tp (width_sort_idx, :);
rendering_image_weight = [0.85, 0.15];


% Original images
subplot(221);
imagesc(im); axis equal;axis tight;axis off; 

% True positives
subplot(222);
[result_im, clipped_bounding_box, text_template, text_tuples] = ...
    dwot_visualize_formatted_bounding_box(im,...
            detectors, formatted_bounding_boxes_tp(width_sort_idx,:), color_range, text_mode,...
            rendering_image_weight, color_map, draw_padding);
imagesc(result_im); axis equal; axis tight; axis off;

% Draw True Positives with bounding box annotations
subplot(223);
imagesc(result_im); axis equal; axis tight; axis off;
dwot_visualize_bounding_boxes(clipped_bounding_box, formatted_bounding_boxes_tp(width_sort_idx,end),...
            text_template, text_tuples, color_range, color_map)

% False positives
subplot(224);
fp_idx = find(false_positive);
fp_idx = fp_idx(1:min(numel(fp_idx), n_max_fals_positive_visualization));
dwot_visualize_formatted_bounding_box(im,...
            detectors, formatted_bounding_boxes(fp_idx,:), color_range, text_mode,...
            rendering_image_weight, color_map, draw_padding);

drawnow;
spaceplots();
drawnow;
