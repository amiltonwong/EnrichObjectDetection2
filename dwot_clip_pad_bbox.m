function clip_padded_bbox = dwot_clip_pad_bbox(bbox, padding_ratio, image_size)

width  = bbox(3) - bbox(1);
height = bbox(4) - bbox(2);
pad_width  = ceil(padding_ratio * width);
pad_height = ceil(padding_ratio * height);
padded_bbox = [bbox(1)-pad_width, bbox(2)-pad_height,...
               bbox(3)+pad_width, bbox(4)+pad_height];

clip_padded_bbox = dwot_clip_bounding_box(padded_bbox, image_size);
