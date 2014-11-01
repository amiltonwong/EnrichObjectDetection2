function null_padded_box = dwot_return_null_padded_box(formatted_bounding_box, score_threshold, box_col_size)
% Given a formatted box with box(:,end) is score, return null box if the input box is empty.
if numel(formatted_bounding_box) == 0 
    null_box = zeros(1,box_col_size);
    null_box( end ) = -inf;
end
