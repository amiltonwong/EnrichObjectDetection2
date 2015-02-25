function [extracted_im, clip_padded_bbox, clip_padded_bbox_offset, width, height] =...
    dwot_extract_image_with_padding(im, bbox, extraction_padding_ratio, im_size)
% box = [ x1 y1 x2 y2]
if nargin < 4
    im_size = size(im);
end

clip_padded_bbox = dwot_clip_pad_bbox(bbox, extraction_padding_ratio, im_size);
clip_padded_bbox_offset = [clip_padded_bbox(1:2) clip_padded_bbox(1:2)]; % x and y coordinate

width  = clip_padded_bbox(3)-clip_padded_bbox(1);
height = clip_padded_bbox(4)-clip_padded_bbox(2);

% bbox_clip = clip_to_image(round(bbox), [1 1 im_size(2) im_size(1)]);
extracted_im = im(clip_padded_bbox(2):clip_padded_bbox(4),...
                  clip_padded_bbox(1):clip_padded_bbox(3), :);
