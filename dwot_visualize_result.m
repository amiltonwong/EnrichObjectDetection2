% figure(2);
nDet = size(bbsNMS,1);
if nDet > 0
    bbsNMS(:,9) = bbsNMS_clip(:,9);
end

tpIdx = logical(tp{imgIdx});
% tpIdx = bbsNMS(:, 9) > param.min_overlap;

% Original images
subplot(221);
imagesc(im); axis off; axis equal;

% True positives
subplot(222);
result_im = dwot_draw_overlap_rendering(im, bbsNMS(tpIdx,:), detectors,...
                                        1, 50, false, [0.15, 0.85, 0], color_range );
imagesc(result_im); axis off; axis equal;
% dwot_draw_overlap_detection(im, bbsNMS(tpIdx,:), renderings, depth_mask, 1, 50, visualize_detection, [0.3, 0.7, 0], color_range );

subplot(223);
dwot_draw_overlap_rendering(im, bbsNMS(tpIdx,:), detectors, inf, 50,...
                            visualize_detection, [0.15, 0.85, 0], color_range );
% dwot_draw_overlap_detection(im, bbsNMS(tpIdx,:), renderings, inf, 50, visualize_detection, [0.3, 0.5, 0.2], color_range );

% False positives
subplot(224);
dwot_draw_overlap_rendering(im, bbsNMS(~tpIdx,:), detectors, 5, 50,...
                            visualize_detection, [0.1, 0.9, 0], color_range );
% dwot_draw_overlap_detection(im, bbsNMS(~tpIdx,:), renderings, 5, 50, visualize_detection, [0.3, 0.7, 0], color_range );

drawnow;
spaceplots();

drawnow;

