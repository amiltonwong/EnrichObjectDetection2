function dwot_visualize_proposal_tuning(bbsNMS, bbsProposal, best_proposal, im, detectors, param)

imSz = size(im);
% clf;
% Plot original image with GT bounding box
subplot(221);
imagesc(im);
%     rectangle('position',dwot_bbox_xy_to_wh(GT_bbox),'edgecolor',[0.7 0.7 0.7],'LineWidth',3);
%     rectangle('position',dwot_bbox_xy_to_wh(GT_bbox),'edgecolor',[0   0   0.6],'LineWidth',2);
axis equal; axis tight;

% Plot proposal bbox 
subplot(222);
dwot_draw_overlap_rendering(im, bbsNMS, detectors, 1, 50, true, [0.1, 0.9, 0] , param.color_range );
axis equal; axis tight;

% Plot tuned bbox
subplot(223);
dwot_draw_overlap_rendering(im, bbsProposal, {best_proposal}, 1, 50, true, [0.1, 0.9, 0] , param.color_range );
axis equal; axis tight;

subplot(224);
cla;

drawnow;
spaceplots();
drawnow;
