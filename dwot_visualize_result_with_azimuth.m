function dwot_visualize_result_with_azimuth(im, formatted_bounding_box, tp_logical, ground_truth_bounding_box, ground_truth_azimuth, detectors, color_range)

n_ground_truth = numel(ground_truth_azimuth);

subplot(221);
imagesc(im);axis equal; axis off; axis tight;

subplot(222);
imagesc(im);
for draw_gt_idx = 1:n_ground_truth
    box_text = sprintf('id:%d a:%0.2f', draw_gt_idx, ground_truth_azimuth(draw_gt_idx));
    current_draw_gt_bbox = ground_truth_bounding_box(draw_gt_idx,:);
    rectangle('position', dwot_bbox_xy_to_wh(current_draw_gt_bbox),'edgecolor',[0.5 0.5 0.5],'LineWidth',3);
    text(current_draw_gt_bbox(1) + 1 , current_draw_gt_bbox(2), box_text, 'BackgroundColor','w','EdgeColor',[0.5 0.5 0.5],'VerticalAlignment','bottom');
end
axis equal; axis off; axis tight;

subplot(223);
dwot_draw_overlap_rendering(im, formatted_bounding_box(tp_logical, :),...
    detectors, 5, 0, true, [0.2, 0.8, 0], color_range, 3 );

subplot(224);
dwot_draw_overlap_rendering(im, formatted_bounding_box(~tp_logical, :),...
    detectors, 5, 0, true, [0.2, 0.8, 0], color_range, 3 );

drawnow
spaceplots();