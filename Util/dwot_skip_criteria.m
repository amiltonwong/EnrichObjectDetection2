function [b_skip_im, object_idx] = dwot_skip_criteria(object_annotations, criteria)
if nargin < 2
    error('Wrong number of inputs');
end

n_object  = numel(object_annotations);
b_skip    = false(1, n_object);

for i = 1:n_object
    cur_object_annotation = object_annotations(i);
    b_curr_skip = false;
    for criterion = criteria
      switch criterion{1}
        case 'truncated'
            b_curr_skip  = b_curr_skip || cur_object_annotation.truncated;
        case 'difficult'    
            b_curr_skip  = b_curr_skip || cur_object_annotation.difficult;
        case 'occluded'
            b_curr_skip  = b_curr_skip || cur_object_annotation.occluded;      
        case 'none'
            b_curr_skip  = b_curr_skip || cur_object_annotation.difficult;      
        case 'empty'
            % Nothing
        otherwise
          error('undefined criterion');
      end
    end
    b_skip(i) = b_curr_skip;
end

object_idx = ~b_skip;
if ~ismember('none', criteria)
    b_skip_im = nnz(object_idx) == 0;
else
    b_skip_im = false;
end

if n_object == 0 && ismember('empty', criteria)
    b_skip_im = true;
end