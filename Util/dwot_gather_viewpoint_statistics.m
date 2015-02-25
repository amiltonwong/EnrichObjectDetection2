function [tp_per_template, fp_per_template] = dwot_gather_viewpoint_statistics(tp_per_template, fp_per_template, bbsNMS, min_overlap)

N_DETECTION = size(bbsNMS,1);

for det_idx = 1:N_DETECTION
  template_id = bbsNMS(det_idx,11);
  det_score = bbsNMS(det_idx,12);
  det_overlap = bbsNMS(det_idx,9);
  if det_overlap > min_overlap
    tp_per_template{template_id} = [tp_per_template{template_id} det_score];
  else
    fp_per_template{template_id} = [fp_per_template{template_id} det_score];
  end
end
