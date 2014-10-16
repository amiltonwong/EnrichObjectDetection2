function rect_bbox = dwot_bbox_xy_to_wh(xy_bbox)
    rect_bbox = xy_bbox(:,1:4) - [0 0 x_bbox(:,1:2)];
