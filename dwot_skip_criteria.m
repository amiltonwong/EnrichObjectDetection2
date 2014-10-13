function b_skip = dwot_skip_criteria(object_annotations, criteria)
  if nargin < 2
    error('Wrong number of inputs');
  end
  
  b_skip = false;
  n_annotations = numel(object_annotations);
  for i = 1:n_annotations
    cur_object_annotation = object_annotations(i);
    
    for criterion = criteria
      switch criterion{1}
        case 'truncated'
          b_skip  = b_skip || cur_object_annotation.truncated;
        case 'difficult'    
          b_skip  = b_skip || cur_object_annotation.difficult;
        case 'occluded'
          b_skip  = b_skip || cur_object_annotation.occluded;      
        otherwise
          continue;
      end
    end
  end
  
  if n_annotations == 0 && ismember('empty', criteria)
    b_skip = true;
  end