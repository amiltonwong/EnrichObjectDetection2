nDet = size(bbsNMS,1);
bbsNMSDraw = bbsNMS;
if nDet > 0
    bbsNMSDraw(:,9) = bbsNMS_clip(:,9);
end

% Sort the image according to it width
% the dwot_draw_overlap_rendering will draw from the last box so put the largest object
% first
widths = bbsNMSDraw(:,3) - bbsNMSDraw(:,1);
[~, width_sort_idx ] = sort(widths, 'descend');
bbsNMSDraw = bbsNMSDraw(width_sort_idx, :);
tpIdxSort = tpIdx(width_sort_idx);

% Original images
subplot(221);
imagesc(im); axis off; axis equal;

% True positives
subplot(222);
result_im = dwot_draw_overlap_rendering(im, bbsNMSDraw(tpIdxSort,:), detectors,...
                                        inf, 50, false, [0.15, 0.85, 0], param.color_range  );
imagesc(result_im); axis off; axis equal;

% Draw True Positives with bounding box annotations
subplot(223);
dwot_draw_overlap_rendering(im, bbsNMSDraw(tpIdxSort,:), detectors, inf, 50,...
                            visualize_detection, [0.2, 0.8, 0], param.color_range  );

% False positives
subplot(224);
dwot_draw_overlap_rendering(im, bbsNMS(~tpIdxSort,:), detectors, 5, 50,...
                            visualize_detection, [0.2, 0.8, 0], param.color_range  );

drawnow;
spaceplots();

drawnow;
